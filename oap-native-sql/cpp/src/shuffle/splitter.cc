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

#include "splitter.h"

namespace sparkcolumnarplugin {
namespace shuffle {

// ----------------------------------------------------------------------
// Splitter

arrow::Result<std::shared_ptr<Splitter>> Splitter::Make(
    const std::string& short_name, std::shared_ptr<arrow::Schema> schema,
    int num_partitions, gandiva::ExpressionVector expr_vector,
    gandiva::FieldVector field_vector) {
  if (short_name == "hash") {
    return HashSplitter::Create(num_partitions, std::move(schema), std::move(expr_vector),
                                std::move(field_vector));
  } else if (short_name == "rr") {
    return RoundRobinSplitter::Create(num_partitions, std::move(schema));
  } else if (short_name == "range") {
    return FallbackRangeSplitter::Create(num_partitions, std::move(schema));
  } else if (short_name == "single") {
    return SingleSplitter::Create(std::move(schema));
  } else {
    return arrow::Status::NotImplemented("Partitioning " + short_name +
                                         " not supported yet.");
  }
}

arrow::Result<std::shared_ptr<Splitter>> Splitter::Make(
    const std::string& short_name, std::shared_ptr<arrow::Schema> schema,
    int num_partitions) {
  return Make(short_name, std::move(schema), num_partitions, {}, {});
}

Splitter::~Splitter() = default;

// ----------------------------------------------------------------------
// SingleSplitter

SingleSplitter::SingleSplitter(std::shared_ptr<arrow::Schema> schema,
                               std::string output_file_path)
    : Splitter(std::move(schema)), file_path_(std::move(output_file_path)) {}

arrow::Result<std::shared_ptr<SingleSplitter>> SingleSplitter::Create(
    std::shared_ptr<arrow::Schema> schema) {
  ARROW_ASSIGN_OR_RAISE(auto configured_dirs, GetConfiguredLocalDirs());

  // Pick a configured local dir randomly
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> distrib(0, configured_dirs.size() - 1);
  int pos = distrib(gen);

  ARROW_ASSIGN_OR_RAISE(auto dir, CreateRandomSubDir(configured_dirs[pos]))
  auto file_path = arrow::fs::internal::ConcatAbstractPath(dir, "data");

  std::shared_ptr<SingleSplitter> splitter(
      new SingleSplitter(std::move(schema), std::move(file_path)));

  return splitter;
}

arrow::Status SingleSplitter::Split(const arrow::RecordBatch& rb) {
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
    ARROW_ASSIGN_OR_RAISE(total_bytes_written_, file_os_->Tell());
    file_os_->Close();
  }
  if (file_writer_opened_) {
    file_writer_->Close();
  }
  partition_file_info_ =
      std::vector<std::pair<int32_t, std::string>>{std::make_pair(0, file_path_)};
}

// ----------------------------------------------------------------------
// BasePartitionSplitter

arrow::Status BasePartitionSplitter::Init() {
  const auto& fields = schema_->fields();
  ARROW_ASSIGN_OR_RAISE(column_type_id_, ToSplitterTypeId(schema_->fields()));

  std::vector<Type::typeId> remove_null_id;
  remove_null_id.reserve(column_type_id_.size());
  std::copy_if(std::cbegin(column_type_id_), std::cend(column_type_id_),
               std::back_inserter(remove_null_id),
               [](Type::typeId id) { return id != Type::typeId::SHUFFLE_NULL; });
  last_type_id_ =
      *std::max_element(std::cbegin(remove_null_id), std::cend(remove_null_id));

  ARROW_ASSIGN_OR_RAISE(configured_dirs_, GetConfiguredLocalDirs())

  partition_writer_.resize(num_partitions_);

  return arrow::Status::OK();
}

arrow::Status BasePartitionSplitter::DoSplit(
    const arrow::RecordBatch& rb, std::vector<std::shared_ptr<PartitionWriter>> writers) {
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

  auto read_offset = 0;

#define WRITE_FIXEDWIDTH(TYPE_ID, T)                                             \
  if (!src_addr[TYPE_ID].empty()) {                                              \
    for (i = read_offset; i < num_rows; ++i) {                                   \
      ARROW_ASSIGN_OR_RAISE(auto result,                                         \
                            writers[i]->Write<T>(TYPE_ID, src_addr[TYPE_ID], i)) \
      if (!result) {                                                             \
        break;                                                                   \
      }                                                                          \
    }                                                                            \
  }

#define WRITE_BINARY(func, T, src_arr)                                 \
  if (!src_arr.empty()) {                                              \
    for (i = read_offset; i < num_rows; ++i) {                         \
      ARROW_ASSIGN_OR_RAISE(auto result, writers[i]->func(src_arr, i)) \
      if (!result) {                                                   \
        break;                                                         \
      }                                                                \
    }                                                                  \
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
  int64_t total_bytes = 0;
  int64_t total_time = 0;
  for (const auto& writer : partition_writer_) {
    if (writer != nullptr) {
      RETURN_NOT_OK(writer->Stop());
      ARROW_ASSIGN_OR_RAISE(auto b, writer->GetBytesWritten());
      total_bytes += b;
      total_time += writer->GetWriteTime();
    }
  }
  std::sort(std::begin(partition_file_info_), std::end(partition_file_info_));
  return arrow::Status::OK();
}

arrow::Result<std::string> BasePartitionSplitter::CreateDataFile() {
  int m = configured_dirs_.size();
  ARROW_ASSIGN_OR_RAISE(auto dir, CreateRandomSubDir(configured_dirs_[dir_selection_]))
  dir_selection_ = (dir_selection_ + 1) % m;
  return arrow::fs::internal::ConcatAbstractPath(dir, "data");
}

arrow::Status BasePartitionSplitter::Split(const arrow::RecordBatch& rb) {
  ARROW_ASSIGN_OR_RAISE(auto writers, GetNextBatchPartitionWriter(rb));
  RETURN_NOT_OK(DoSplit(rb, std::move(writers)));
  return arrow::Status::OK();
}

// ----------------------------------------------------------------------
// RoundRobinSplitter

arrow::Result<std::shared_ptr<RoundRobinSplitter>> RoundRobinSplitter::Create(
    int32_t num_partitions, std::shared_ptr<arrow::Schema> schema) {
  std::shared_ptr<RoundRobinSplitter> res(
      new RoundRobinSplitter(num_partitions, std::move(schema)));
  RETURN_NOT_OK(res->Init());
  return res;
}

arrow::Result<std::vector<std::shared_ptr<PartitionWriter>>>
RoundRobinSplitter::GetNextBatchPartitionWriter(const arrow::RecordBatch& rb) {
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
HashSplitter::GetNextBatchPartitionWriter(const arrow::RecordBatch& rb) {
  arrow::ArrayVector outputs;
  RETURN_NOT_OK(projector_->Evaluate(rb, arrow::default_memory_pool(), &outputs));
  auto pid_array = outputs.at(0);
  auto num_rows = rb.num_rows();
  ARROW_CHECK_EQ(pid_array->length(), num_rows);

  std::vector<std::shared_ptr<PartitionWriter>> res;
  res.reserve(num_rows);
  auto pid_arr = reinterpret_cast<const int32_t*>(pid_array->data()->buffers[1]->data());
  for (auto i = 0; i < num_rows; ++i) {
    // positive mod
    auto pid = (pid_arr[i] % num_partitions_ + num_partitions_) % num_partitions_;
    if (partition_writer_[pid] == nullptr) {
      ARROW_ASSIGN_OR_RAISE(auto file_path, CreateDataFile())
      partition_file_info_.emplace_back(pid, std::move(file_path));

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
    int32_t num_partitions, std::shared_ptr<arrow::Schema> schema,
    gandiva::ExpressionVector expr_vector, gandiva::FieldVector field_vector) {
  std::shared_ptr<HashSplitter> res(new HashSplitter(num_partitions, std::move(schema),
                                                     std::move(expr_vector),
                                                     std::move(field_vector)));
  RETURN_NOT_OK(res->Init());
  return res;
}

arrow::Status HashSplitter::CreateProjector() {
  // same seed as spark's
  auto seed = gandiva::TreeExprBuilder::MakeLiteral((int32_t)42);
  gandiva::NodePtr node = seed;
  expr_vector_.reserve(field_vector_.size());

  for (const auto& field : field_vector_) {
    if (!field->type()->Equals(arrow::null())) {
      auto field_ptr = gandiva::TreeExprBuilder::MakeField(field);
      node = gandiva::TreeExprBuilder::MakeFunction("hash32", {field_ptr, node},
                                                    arrow::int32());
      auto expr = gandiva::TreeExprBuilder::MakeExpression(
          node, arrow::field("pid", arrow::int32()));
      expr_vector_.push_back(expr);
    }
  }
  RETURN_NOT_OK(gandiva::Projector::Make(schema_, expr_vector_, &projector_));
  return arrow::Status::OK();
}

arrow::Result<std::shared_ptr<FallbackRangeSplitter>> FallbackRangeSplitter::Create(
    int32_t num_partitions, std::shared_ptr<arrow::Schema> schema) {
  auto res = std::shared_ptr<FallbackRangeSplitter>(
      new FallbackRangeSplitter(num_partitions, std::move(schema)));
  RETURN_NOT_OK(res->Init());
  return res;
}

arrow::Status FallbackRangeSplitter::Init() {
  input_schema_ = std::move(schema_);
  ARROW_ASSIGN_OR_RAISE(schema_, input_schema_->RemoveField(0))
  RETURN_NOT_OK(BasePartitionSplitter::Init());
  return arrow::Status::OK();
}

arrow::Status FallbackRangeSplitter::Split(const arrow::RecordBatch& rb) {
  ARROW_ASSIGN_OR_RAISE(auto writers, GetNextBatchPartitionWriter(rb));
  ARROW_ASSIGN_OR_RAISE(auto remove_pid, rb.RemoveColumn(0));
  RETURN_NOT_OK(DoSplit(*remove_pid, std::move(writers)));
  return arrow::Status::OK();
}

arrow::Result<std::vector<std::shared_ptr<PartitionWriter>>>
FallbackRangeSplitter::GetNextBatchPartitionWriter(const arrow::RecordBatch& rb) {
  auto num_rows = rb.num_rows();
  std::vector<std::shared_ptr<PartitionWriter>> res;
  res.reserve(num_rows);

  ARROW_CHECK_EQ(rb.column(0)->type_id(), arrow::Type::INT32);
  auto pid_arr = reinterpret_cast<const int32_t*>(rb.column_data(0)->buffers[1]->data());
  for (auto i = 0; i < num_rows; ++i) {
    auto pid = pid_arr[i];
    ARROW_CHECK_LT(pid, num_partitions_);
    if (partition_writer_[pid] == nullptr) {
      ARROW_ASSIGN_OR_RAISE(auto file_path, CreateDataFile())
      partition_file_info_.emplace_back(pid, std::move(file_path));

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

}  // namespace shuffle
}  // namespace sparkcolumnarplugin
