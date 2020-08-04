/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <gandiva/projector.h>
#include <gandiva/tree_expr_builder.h>
#include <iostream>
#include <memory>
#include <utility>

#include "partition_splitter.h"

namespace sparkcolumnarplugin {
namespace shuffle {

// ----------------------------------------------------------------------
// Splitter

arrow::Result<std::shared_ptr<Splitter>> Splitter::Make(
    const std::string& short_name, std::shared_ptr<arrow::Schema> schema,
    int num_partitions, int32_t buffer_size, arrow::Compression::type compression_type,
    gandiva::ExpressionVector expr_vector, gandiva::FieldVector field_vector) {
  if (short_name == "single") {
    return SingleSplitter::Create(std::move(schema), compression_type);
  } else if (short_name == "rr") {
    return RoundRobinSplitter::Create(num_partitions, std::move(schema), buffer_size,
                                      compression_type);
  } else if (short_name == "hash") {
    return HashSplitter::Create(num_partitions, std::move(schema), buffer_size,
                                compression_type, std::move(expr_vector),
                                std::move(field_vector));
  } else {
    return arrow::Status::NotImplemented("Partitioning " + short_name +
                                         " not supported yet.");
  }
}

Splitter::~Splitter() = default;

// ----------------------------------------------------------------------
// SingleSplitter

SingleSplitter::SingleSplitter(std::shared_ptr<arrow::Schema> schema,
                               arrow::Compression::type compression_codec,
                               std::string output_file_path)
    : Splitter(std::move(schema), compression_codec),
      file_path_(std::move(output_file_path)) {}

arrow::Result<std::shared_ptr<SingleSplitter>> SingleSplitter::Create(
    std::shared_ptr<arrow::Schema> schema, arrow::Compression::type compression_type) {
  ARROW_ASSIGN_OR_RAISE(auto configured_dirs, GetConfiguredLocalDirs());

  // Pick a configured local dir randomly
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> distrib(0, configured_dirs.size() - 1);
  int pos = distrib(gen);

  ARROW_ASSIGN_OR_RAISE(auto dir, CreateRandomSubDir(configured_dirs[pos]))
  auto file_path = arrow::fs::internal::ConcatAbstractPath(dir, "data");

  std::shared_ptr<SingleSplitter> splitter(
      new SingleSplitter(std::move(schema), compression_type, std::move(file_path)));

  return splitter;
}

arrow::Status SingleSplitter::Split(const arrow::RecordBatch& rb) {
  ARROW_CHECK(rb.schema()->Equals(schema_))
      << "RecordBatch schema doesn't match, expected:\n"
      << schema_->ToString() << "\n\n"
      << "actual:\n"
      << rb.schema()->ToString() << "\n";

  if (!file_os_opened_) {
    ARROW_ASSIGN_OR_RAISE(file_os_, arrow::io::FileOutputStream::Open(file_path_, true))
    file_os_opened_ = true;
  }
  if (!file_writer_opened_) {
    ARROW_ASSIGN_OR_RAISE(
        file_writer_, arrow::ipc::NewStreamWriter(file_os_.get(), schema_,
                                                  GetIpcWriteOptions(compression_type_)))
    file_writer_opened_ = true;
  }

  TIME_MICRO_OR_RAISE(total_write_time_, file_writer_->WriteRecordBatch(rb));
  return arrow::Status::OK();
}

arrow::Status SingleSplitter::Stop() {
  if (!file_os_->closed()) {
    ARROW_ASSIGN_OR_RAISE(bytes_written_, file_os_->Tell());
    file_os_->Close();
  }
  if (file_writer_opened_) {
    file_writer_->Close();
  }
  partition_file_info_ =
      std::vector<std::pair<int32_t, std::string>>{std::make_pair(0, file_path_)};
}

arrow::Result<int64_t> SingleSplitter::TotalBytesWritten() {
  if (!file_os_->closed()) {
    ARROW_ASSIGN_OR_RAISE(bytes_written_, file_os_->Tell());
  }
  return bytes_written_;
}

int64_t SingleSplitter::TotalWriteTime() { return total_write_time_; }

// ----------------------------------------------------------------------
// BasePartitionSplitter

arrow::Status BasePartitionSplitter::Init() {
  const auto& fields = schema_->fields();
  std::vector<Type::typeId> result;
  result.reserve(fields.size());
  std::pair<std::string, arrow::Type::type> field_type_not_implemented;

  std::transform(std::cbegin(fields), std::cend(fields), std::back_inserter(result),
                 [&field_type_not_implemented](
                     const std::shared_ptr<arrow::Field>& field) -> Type::typeId {
                   auto arrow_type_id = field->type()->id();
                   switch (arrow_type_id) {
                     case arrow::BooleanType::type_id:
                       return Type::SHUFFLE_BIT;
                     case arrow::Int8Type::type_id:
                     case arrow::UInt8Type::type_id:
                       return Type::SHUFFLE_1BYTE;
                     case arrow::Int16Type::type_id:
                     case arrow::UInt16Type::type_id:
                     case arrow::HalfFloatType::type_id:
                       return Type::SHUFFLE_2BYTE;
                     case arrow::Int32Type::type_id:
                     case arrow::UInt32Type::type_id:
                     case arrow::FloatType::type_id:
                     case arrow::Date32Type::type_id:
                     case arrow::Time32Type::type_id:
                       return Type::SHUFFLE_4BYTE;
                     case arrow::Int64Type::type_id:
                     case arrow::UInt64Type::type_id:
                     case arrow::DoubleType::type_id:
                     case arrow::Date64Type::type_id:
                     case arrow::Time64Type::type_id:
                     case arrow::TimestampType::type_id:
                       return Type::SHUFFLE_8BYTE;
                     case arrow::BinaryType::type_id:
                     case arrow::StringType::type_id:
                       return Type::SHUFFLE_BINARY;
                     case arrow::LargeBinaryType::type_id:
                     case arrow::LargeStringType::type_id:
                       return Type::SHUFFLE_LARGE_BINARY;
                     case arrow::NullType::type_id:
                       return Type::SHUFFLE_NULL;
                     default:
                       field_type_not_implemented =
                           std::make_pair(std::move(field->ToString()), arrow_type_id);
                       return Type::SHUFFLE_NOT_IMPLEMENTED;
                   }
                 });

  auto it =
      std::find(std::begin(result), std::end(result), Type::SHUFFLE_NOT_IMPLEMENTED);
  if (it != std::end(result)) {
    RETURN_NOT_OK(arrow::Status::NotImplemented(
        "Field type not implemented: " + field_type_not_implemented.first +
        "\n arrow type id: " + std::to_string(field_type_not_implemented.second)));
  }
  column_type_id_ = std::move(result);

  decltype(column_type_id_) remove_null_id(column_type_id_.size());
  std::copy_if(std::cbegin(column_type_id_), std::cend(column_type_id_),
               std::begin(remove_null_id),
               [](Type::typeId id) { return id != Type::typeId::SHUFFLE_NULL; });
  last_type_id_ =
      *std::max_element(std::cbegin(remove_null_id), std::cend(remove_null_id));

  ARROW_ASSIGN_OR_RAISE(configured_dirs_, GetConfiguredLocalDirs())

  partition_writer_.resize(num_partitions_);

  return arrow::Status::OK();
}

arrow::Status BasePartitionSplitter::Split(const arrow::RecordBatch& rb) {
  auto num_rows = rb.num_rows();
  auto num_cols = rb.num_columns();
  auto src_addr = std::vector<SrcBuffers>(Type::NUM_TYPES);

  auto src_binary_arr = SrcBinaryArrays();
  auto src_nullable_binary_arr = SrcBinaryArrays();

  auto src_large_binary_arr = SrcLargeBinaryArrays();
  auto src_nullable_large_binary_arr = SrcLargeBinaryArrays();

  arrow::TypedBufferBuilder<bool> null_bitmap_builder_;
  RETURN_NOT_OK(null_bitmap_builder_.Append(num_rows, true));

  std::shared_ptr<arrow::Buffer> dummy_buf;
  RETURN_NOT_OK(null_bitmap_builder_.Finish(&dummy_buf));
  auto dummy_buf_p = const_cast<uint8_t*>(dummy_buf->data());

  // Get the pointer of each buffer id
  for (auto i = 0; i < num_cols; ++i) {
    const auto& buffers = rb.column_data(i)->buffers;
    if (rb.column_data(i)->GetNullCount() == 0) {
      if (column_type_id_[i] == Type::SHUFFLE_BINARY) {
        src_binary_arr.push_back(
            std::static_pointer_cast<arrow::BinaryArray>(rb.column(i)));
      } else if (column_type_id_[i] == Type::SHUFFLE_LARGE_BINARY) {
        src_large_binary_arr.push_back(
            std::static_pointer_cast<arrow::LargeBinaryArray>(rb.column(i)));
      } else if (column_type_id_[i] != Type::SHUFFLE_NULL) {
        // null bitmap may be nullptr
        src_addr[column_type_id_[i]].push_back(
            {.validity_addr = dummy_buf_p,
             .value_addr = const_cast<uint8_t*>(buffers[1]->data())});
      }
    } else {
      if (column_type_id_[i] == Type::SHUFFLE_BINARY) {
        src_nullable_binary_arr.push_back(
            std::static_pointer_cast<arrow::BinaryArray>(rb.column(i)));
      } else if (column_type_id_[i] == Type::SHUFFLE_LARGE_BINARY) {
        src_nullable_large_binary_arr.push_back(
            std::static_pointer_cast<arrow::LargeBinaryArray>(rb.column(i)));
      } else if (column_type_id_[i] != Type::SHUFFLE_NULL) {
        src_addr[column_type_id_[i]].push_back(
            {.validity_addr = const_cast<uint8_t*>(buffers[0]->data()),
             .value_addr = const_cast<uint8_t*>(buffers[1]->data())});
      }
    }
  }

  ARROW_ASSIGN_OR_RAISE(auto writer, GetPartitionWriter(rb));

  auto read_offset = 0;

#define WRITE_FIXEDWIDTH(TYPE_ID, T)                                            \
  if (!src_addr[TYPE_ID].empty()) {                                             \
    for (i = read_offset; i < num_rows; ++i) {                                  \
      ARROW_ASSIGN_OR_RAISE(auto result,                                        \
                            writer[i]->Write<T>(TYPE_ID, src_addr[TYPE_ID], i)) \
      if (!result) {                                                            \
        break;                                                                  \
      }                                                                         \
    }                                                                           \
  }

#define WRITE_BINARY(func, T, src_arr)                                \
  if (!src_arr.empty()) {                                             \
    for (i = read_offset; i < num_rows; ++i) {                        \
      ARROW_ASSIGN_OR_RAISE(auto result, writer[i]->func(src_arr, i)) \
      if (!result) {                                                  \
        break;                                                        \
      }                                                               \
    }                                                                 \
  }

  while (read_offset < num_rows) {
    auto i = read_offset;
    WRITE_FIXEDWIDTH(Type::SHUFFLE_1BYTE, uint8_t);
    WRITE_FIXEDWIDTH(Type::SHUFFLE_2BYTE, uint16_t);
    WRITE_FIXEDWIDTH(Type::SHUFFLE_4BYTE, uint32_t);
    WRITE_FIXEDWIDTH(Type::SHUFFLE_8BYTE, uint64_t);
    WRITE_FIXEDWIDTH(Type::SHUFFLE_BIT, bool);
    WRITE_BINARY(WriteBinary, arrow::BinaryType, src_binary_arr);
    WRITE_BINARY(WriteLargeBinary, arrow::LargeBinaryType, src_large_binary_arr);
    WRITE_BINARY(WriteNullableBinary, arrow::BinaryType, src_nullable_binary_arr);
    WRITE_BINARY(WriteNullableLargeBinary, arrow::LargeBinaryType,
                 src_nullable_large_binary_arr);
    read_offset = i;
  }

#undef WRITE_FIXEDWIDTH
#undef WRITE_BINARY

  return arrow::Status::OK();
}

arrow::Status BasePartitionSplitter::Stop() {
  for (const auto& writer : partition_writer_) {
    RETURN_NOT_OK(writer->Stop());
  }
  std::sort(std::begin(partition_file_info_), std::end(partition_file_info_));
  return arrow::Status::OK();
}

arrow::Result<int64_t> BasePartitionSplitter::TotalBytesWritten() {
  int64_t res = 0;
  for (const auto& writer : partition_writer_) {
    ARROW_ASSIGN_OR_RAISE(auto bytes, writer->BytesWritten());
    res += bytes;
  }
  return res;
}

int64_t BasePartitionSplitter::TotalWriteTime() {
  uint64_t res = 0;
  for (const auto& writer : partition_writer_) {
    res += writer->write_time();
  }
  return res;
}

arrow::Result<std::string> BasePartitionSplitter::CreateDataFile() {
  int m = configured_dirs_.size();
  ARROW_ASSIGN_OR_RAISE(auto dir, CreateRandomSubDir(configured_dirs_[dir_selection_]))
  dir_selection_ = (dir_selection_ + 1) % m;
  return arrow::fs::internal::ConcatAbstractPath(dir, "data");
}

// ----------------------------------------------------------------------
// RoundRobinSplitter

arrow::Result<std::shared_ptr<RoundRobinSplitter>> RoundRobinSplitter::Create(
    int32_t num_partitions, std::shared_ptr<arrow::Schema> schema, int32_t buffer_size,
    arrow::Compression::type compression_type) {
  std::shared_ptr<RoundRobinSplitter> res(new RoundRobinSplitter(
      num_partitions, std::move(schema), buffer_size, compression_type));
  RETURN_NOT_OK(res->Init());
  return res;
}

arrow::Result<std::vector<std::shared_ptr<PartitionWriter>>>
RoundRobinSplitter::GetPartitionWriter(const arrow::RecordBatch& rb) {
  auto num_rows = rb.num_rows();

  std::vector<std::shared_ptr<PartitionWriter>> res;
  res.reserve(num_rows);
  for (auto i = 0; i < num_rows; ++i) {
    if (partition_writer_[pid_selection_] == nullptr) {
      ARROW_ASSIGN_OR_RAISE(auto file_path, CreateDataFile())
      partition_file_info_.push_back({pid_selection_, std::move(file_path)});

      ARROW_ASSIGN_OR_RAISE(
          partition_writer_[pid_selection_],
          PartitionWriter::Create(pid_selection_, buffer_size_, last_type_id_,
                                  column_type_id_, schema_,
                                  partition_file_info_.back().second, compression_type_));
    }
    res.push_back(partition_writer_[pid_selection_]);
    pid_selection_ = (pid_selection_ + 1) % num_partitions_;
  }
  return res;
}

// ----------------------------------------------------------------------
// HashSplitter

arrow::Result<std::vector<std::shared_ptr<PartitionWriter>>>
HashSplitter::GetPartitionWriter(const arrow::RecordBatch& rb) {
  arrow::ArrayVector outputs;
  RETURN_NOT_OK(projector_->Evaluate(rb, arrow::default_memory_pool(), &outputs));
  auto pid_array = outputs.at(0);
  auto num_rows = rb.num_rows();
  ARROW_CHECK_EQ(pid_array->length(), num_rows);

  std::vector<std::shared_ptr<PartitionWriter>> res;
  res.reserve(num_rows);
  auto pid_arr = reinterpret_cast<const int32_t*>(pid_array->data()->buffers[1]->data());
  for (int64_t i = 0; i < num_rows; ++i) {
    // positive mod
    auto pid = (pid_arr[i] % num_partitions_ + num_partitions_) % num_partitions_;
    if (partition_writer_[pid] == nullptr) {
      ARROW_ASSIGN_OR_RAISE(auto file_path, CreateDataFile())
      partition_file_info_.push_back({pid, std::move(file_path)});

      ARROW_ASSIGN_OR_RAISE(
          partition_writer_[pid],
          PartitionWriter::Create(pid, buffer_size_, last_type_id_, column_type_id_,
                                  schema_, partition_file_info_.back().second,
                                  compression_type_))
    }
    res.push_back(partition_writer_[pid]);
  }
  return res;
}

arrow::Result<std::shared_ptr<HashSplitter>> HashSplitter::Create(
    int32_t num_partitions, std::shared_ptr<arrow::Schema> schema, int32_t buffer_size,
    arrow::Compression::type compression_type, gandiva::ExpressionVector expr_vector,
    gandiva::FieldVector field_vector) {
  std::shared_ptr<HashSplitter> res(
      new HashSplitter(num_partitions, std::move(schema), buffer_size, compression_type,
                       std::move(expr_vector), std::move(field_vector)));
  RETURN_NOT_OK(res->Init());
  return res;
}

arrow::Status HashSplitter::CreateProjector() {
  std::vector<gandiva::NodePtr> fields;
  fields.reserve(field_vector_.size());
  std::transform(
      std::cbegin(field_vector_), std::cend(field_vector_), std::back_inserter(fields),
      [](const std::shared_ptr<arrow::Field>& field) { return gandiva::TreeExprBuilder::MakeField(field); });

  auto exprs = gandiva::TreeExprBuilder::MakeExpression("hash32", {field_vector_},
                                                        field("pid", arrow::int32()));
  expr_vector_.push_back(std::move(exprs));
  RETURN_NOT_OK(gandiva::Projector::Make(schema_, expr_vector_, &projector_));
  return arrow::Status::OK();
}

}  // namespace shuffle
}  // namespace sparkcolumnarplugin
