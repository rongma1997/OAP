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

#include <memory>
#include <utility>

#include <arrow/ipc/writer.h>
#include <arrow/util/bit_util.h>
#include <gandiva/node.h>
#include <gandiva/projector.h>
#include <gandiva/tree_expr_builder.h>

#include "shuffle/splitter.h"
#include "shuffle/utils.h"
#include "utils/macros.h"

namespace sparkcolumnarplugin {
namespace shuffle {

SplitOptions SplitOptions::Defaults() { return SplitOptions(); }

class Splitter::PartitionWriter {
 public:
  explicit PartitionWriter(Splitter* splitter, std::string spilled_file)
      : splitter_(splitter), spilled_file_(std::move(spilled_file)) {}

  arrow::Status Spill(const std::shared_ptr<arrow::RecordBatch>& batch) {
    RETURN_NOT_OK(EnsureOpened());
    TIME_NANO_OR_RAISE(spill_time, spilled_file_writer_->WriteRecordBatch(*batch));
    return arrow::Status::OK();
  }

  arrow::Status WriteLastRecordBatchAndClose(
      const std::shared_ptr<arrow::RecordBatch>& batch) {
    auto start_write = std::chrono::steady_clock::now();
    const auto& data_file_os = splitter_->data_file_os_;
    ARROW_ASSIGN_OR_RAISE(auto before_write, data_file_os->Tell());

    if (spilled_file_opened_) {
      ARROW_ASSIGN_OR_RAISE(
          auto spilled_file_is_,
          arrow::io::MemoryMappedFile::Open(spilled_file_, arrow::io::FileMode::READ));
      // copy spilled data blocks
      ARROW_ASSIGN_OR_RAISE(auto nbytes, spilled_file_is_->GetSize());
      ARROW_ASSIGN_OR_RAISE(auto buffer, spilled_file_is_->Read(nbytes));
      RETURN_NOT_OK(data_file_os->Write(buffer));

      // close spilled file streams and delete the file
      RETURN_NOT_OK(spilled_file_os_->Close());
      RETURN_NOT_OK(spilled_file_is_->Close());
      auto fs = std::make_shared<arrow::fs::LocalFileSystem>();
      RETURN_NOT_OK(fs->DeleteFile(spilled_file_));
      bytes_spilled += nbytes;

      // write last record batch if it's not null
      if (batch != nullptr) {
        int32_t metadata_length;
        int64_t body_length;
        RETURN_NOT_OK(arrow::ipc::WriteRecordBatch(
            *batch, 0, data_file_os.get(), &metadata_length, &body_length,
            SplitterIpcWriteOptions(splitter_->options_.compression_type)));
      }
      // write EOS
      constexpr int32_t kZeroLength = 0;
      RETURN_NOT_OK(data_file_os->Write(&kIpcContinuationToken, sizeof(int32_t)));
      RETURN_NOT_OK(data_file_os->Write(&kZeroLength, sizeof(int32_t)));
    } else {
      ARROW_ASSIGN_OR_RAISE(
          auto data_file_writer,
          arrow::ipc::NewStreamWriter(
              data_file_os.get(), splitter_->schema_,
              SplitterIpcWriteOptions(splitter_->options_.compression_type)));
      // write last record batch, it is the only batch to write so it can't be null
      if (batch == nullptr) {
        return arrow::Status::Invalid("Partition writer got empty partition");
      }
      RETURN_NOT_OK(data_file_writer->WriteRecordBatch(*batch));
      // write EOS
      RETURN_NOT_OK(data_file_writer->Close());
    }

    ARROW_ASSIGN_OR_RAISE(auto after_write, data_file_os->Tell());
    partition_length = after_write - before_write;

    auto end_write = std::chrono::steady_clock::now();
    write_time =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end_write - start_write)
            .count();

    return arrow::Status::OK();
  }

  // metrics
  int64_t spill_time = 0;
  int64_t write_time = 0;
  int64_t bytes_spilled = 0;
  int64_t partition_length = 0;

 private:
  arrow::Status EnsureOpened() {
    if (!spilled_file_opened_) {
      ARROW_ASSIGN_OR_RAISE(spilled_file_os_,
                            arrow::io::FileOutputStream::Open(spilled_file_, true));
      ARROW_ASSIGN_OR_RAISE(
          spilled_file_writer_,
          arrow::ipc::NewStreamWriter(
              spilled_file_os_.get(), splitter_->schema_,
              SplitterIpcWriteOptions(splitter_->options_.compression_type)));
      spilled_file_opened_ = true;
    }
    return arrow::Status::OK();
  }

  Splitter* splitter_;
  std::string spilled_file_;
  std::shared_ptr<arrow::io::FileOutputStream> spilled_file_os_;
  std::shared_ptr<arrow::ipc::RecordBatchWriter> spilled_file_writer_;

  bool spilled_file_opened_ = false;
};

// ----------------------------------------------------------------------
// Splitter

arrow::Result<std::shared_ptr<Splitter>> Splitter::Make(
    const std::string& short_name, std::shared_ptr<arrow::Schema> schema,
    int num_partitions, const gandiva::ExpressionVector& expr_vector,
    SplitOptions options) {
  if (short_name == "hash") {
    return HashSplitter::Create(num_partitions, std::move(schema), expr_vector,
                                std::move(options));
  } else if (short_name == "rr") {
    return RoundRobinSplitter::Create(num_partitions, std::move(schema),
                                      std::move(options));
  } else if (short_name == "range") {
    return FallbackRangeSplitter::Create(num_partitions, std::move(schema),
                                         std::move(options));
  } else if (short_name == "single") {
    return RoundRobinSplitter::Create(1, std::move(schema), std::move(options));
  }
  return arrow::Status::NotImplemented("Partitioning " + short_name +
                                       " not supported yet.");
}

arrow::Result<std::shared_ptr<Splitter>> Splitter::Make(
    const std::string& short_name, std::shared_ptr<arrow::Schema> schema,
    int num_partitions, SplitOptions options) {
  return Make(short_name, std::move(schema), num_partitions, {}, std::move(options));
}

arrow::Status Splitter::Init() {
  const auto& fields = schema_->fields();
  ARROW_ASSIGN_OR_RAISE(column_type_id_, ToSplitterTypeId(schema_->fields()));

  partition_writer_.resize(num_partitions_);
  partition_id_cnt_.resize(num_partitions_);
  partition_buffer_initialized_.resize(num_partitions_);
  partition_buffer_base_.resize(num_partitions_);
  partition_buffer_offset_.resize(num_partitions_);
  partition_lengths_.reserve(num_partitions_);

  for (int i = 0; i < column_type_id_.size(); ++i) {
    switch (column_type_id_[i]) {
      case Type::SHUFFLE_BINARY:
        binary_array_idx_.push_back(i);
        break;
      case Type::SHUFFLE_LARGE_BINARY:
        large_binary_array_idx_.push_back(i);
        break;
      case Type::SHUFFLE_NULL:
        break;
      default:
        fixed_width_array_idx_.push_back(i);
        break;
    }
  }

  auto num_fixed_width = fixed_width_array_idx_.size();
  partition_fixed_width_validity_addrs_.resize(num_fixed_width);
  partition_fixed_width_value_addrs_.resize(num_fixed_width);
  partition_fixed_width_buffers_.resize(num_fixed_width);
  for (auto i = 0; i < num_fixed_width; ++i) {
    partition_fixed_width_validity_addrs_[i].resize(num_partitions_);
    partition_fixed_width_value_addrs_[i].resize(num_partitions_);
    partition_fixed_width_buffers_[i].resize(num_partitions_);
  }
  partition_binary_builders_.resize(binary_array_idx_.size());
  for (auto i = 0; i < binary_array_idx_.size(); ++i) {
    partition_binary_builders_[i].resize(num_partitions_);
  }
  partition_large_binary_builders_.resize(large_binary_array_idx_.size());
  for (auto i = 0; i < large_binary_array_idx_.size(); ++i) {
    partition_large_binary_builders_[i].resize(num_partitions_);
  }

  ARROW_ASSIGN_OR_RAISE(configured_dirs_, GetConfiguredLocalDirs());
  sub_dir_selection_.assign(configured_dirs_.size(), 0);

  // Both data_file and shuffle_index_file should be set through jni.
  // For test purpose, Create a temporary subdirectory in the system temporary dir with
  // prefix "columnar-shuffle"
  if (options_.data_file.length() == 0) {
    ARROW_ASSIGN_OR_RAISE(options_.data_file, CreateTempShuffleFile(configured_dirs_[0]));
  }
  return arrow::Status::OK();
}

arrow::Status Splitter::Split(const arrow::RecordBatch& rb) {
  EVAL_START("split", options_.thread_id)
  RETURN_NOT_OK(ComputeAndCountPartitionId(rb));
  RETURN_NOT_OK(DoSplit(rb));
  EVAL_END("split", options_.thread_id, options_.task_attempt_id)
  return arrow::Status::OK();
}

arrow::Status Splitter::Stop() {
  EVAL_START("write", options_.thread_id)
  // open data file output stream
  ARROW_ASSIGN_OR_RAISE(data_file_os_,
                        arrow::io::FileOutputStream::Open(options_.data_file, true));

  // stop PartitionWriter and collect metrics
  for (auto pid = 0; pid < num_partitions_; ++pid) {
    if (partition_buffer_base_[pid] > 0) {
      if (partition_writer_[pid] == nullptr) {
        RETURN_NOT_OK(CreatePartitionWriter(pid));
      }
      ARROW_ASSIGN_OR_RAISE(auto batch, MakeRecordBatchAndReset(pid));
      RETURN_NOT_OK(partition_writer_[pid]->WriteLastRecordBatchAndClose(batch));
    }
    if (partition_writer_[pid] != nullptr) {
      const auto& writer = partition_writer_[pid];
      partition_lengths_.push_back(writer->partition_length);
      total_bytes_written_ += writer->partition_length;
      total_bytes_spilled_ += writer->bytes_spilled;
      total_write_time_ += writer->write_time;
      total_spill_time_ += writer->spill_time;
    } else {
      partition_lengths_.push_back(0);
    }
  }

  // close data file output Stream
  RETURN_NOT_OK(data_file_os_->Close());

  EVAL_END("write", options_.thread_id, options_.task_attempt_id)
  return arrow::Status::OK();
}

arrow::Status Splitter::InitPartitionBuffers(int32_t partition_id) {
  auto fixed_width_idx = 0;
  auto binary_idx = 0;
  auto large_binary_idx = 0;
  auto num_fields = schema_->num_fields();
  for (auto i = 0; i < num_fields; ++i) {
    switch (column_type_id_[i]) {
      case Type::SHUFFLE_BINARY:
        partition_binary_builders_[binary_idx++][partition_id] =
            std::make_shared<arrow::BinaryBuilder>(options_.memory_pool);
        break;
      case Type::SHUFFLE_LARGE_BINARY:
        partition_large_binary_builders_[large_binary_idx++][partition_id] =
            std::make_shared<arrow::LargeBinaryBuilder>(options_.memory_pool);
        break;
      case Type::SHUFFLE_NULL:
        break;
      default:
        std::shared_ptr<arrow::Buffer> value_buffer;
        // only allocate for value buffer
        if (column_type_id_[i] == Type::SHUFFLE_BIT) {
          ARROW_ASSIGN_OR_RAISE(
              value_buffer,
              arrow::AllocateEmptyBitmap(options_.buffer_size, options_.memory_pool))
        } else {
          ARROW_ASSIGN_OR_RAISE(
              value_buffer,
              arrow::AllocateBuffer(options_.buffer_size * (1 << column_type_id_[i]),
                                    options_.memory_pool))
        }
        partition_fixed_width_value_addrs_[fixed_width_idx][partition_id] =
            const_cast<uint8_t*>(value_buffer->data());
        partition_fixed_width_buffers_[fixed_width_idx++][partition_id] = {
            nullptr, std::move(value_buffer)};
    }
  }
  partition_buffer_initialized_[partition_id] = true;
  return arrow::Status::OK();
}

arrow::Result<std::shared_ptr<arrow::RecordBatch>> Splitter::MakeRecordBatchAndReset(
    int32_t partition_id) {
  auto fixed_width_idx = 0;
  auto binary_idx = 0;
  auto large_binary_idx = 0;
  auto num_fields = schema_->num_fields();
  auto num_rows = partition_buffer_base_[partition_id];
  std::vector<std::shared_ptr<arrow::Array>> arrays(num_fields);
  for (int i = 0; i < num_fields; ++i) {
    switch (column_type_id_[i]) {
      case Type::SHUFFLE_BINARY:
        RETURN_NOT_OK(
            partition_binary_builders_[binary_idx++][partition_id]->Finish(&arrays[i]));
        break;
      case Type::SHUFFLE_LARGE_BINARY:
        RETURN_NOT_OK(
            partition_large_binary_builders_[large_binary_idx++][partition_id]->Finish(
                &arrays[i]));
        break;
      case Type::SHUFFLE_NULL:
        arrays[i] = arrow::MakeArray(arrow::ArrayData::Make(
            arrow::null(), num_rows, {nullptr, nullptr}, num_rows));
        break;
      default:
        arrays[i] = arrow::MakeArray(arrow::ArrayData::Make(
            schema_->field(i)->type(), num_rows,
            partition_fixed_width_buffers_[fixed_width_idx][partition_id]));
        partition_fixed_width_validity_addrs_[fixed_width_idx][partition_id] = nullptr;
        partition_fixed_width_buffers_[fixed_width_idx++][partition_id][0] = nullptr;
        break;
    }
  }
  partition_buffer_base_[partition_id] = 0;
  return arrow::RecordBatch::Make(schema_, num_rows, std::move(arrays));
}

arrow::Status Splitter::DoSplit(const arrow::RecordBatch& rb) {
  // prepare partition buffers and spill if necessary
  for (auto pid = 0; pid < num_partitions_; ++pid) {
    if (partition_id_cnt_[pid] > 0) {
      // if this record batch exceed dst record batch limitation, spill the dst record
      // batch
      if (partition_buffer_base_[pid] + partition_id_cnt_[pid] > options_.buffer_size) {
        if (!partition_buffer_initialized_[pid]) {
          return arrow::Status::Invalid("Partition size exceeds maximum buffer size");
        }
        // spill this partition
        if (partition_writer_[pid] == nullptr) {
          RETURN_NOT_OK(CreatePartitionWriter(pid));
        }
        ARROW_ASSIGN_OR_RAISE(auto batch, MakeRecordBatchAndReset(pid));
        RETURN_NOT_OK(partition_writer_[pid]->Spill(batch));
      } else if (!partition_buffer_initialized_[pid]) {
        RETURN_NOT_OK(InitPartitionBuffers(pid));
      }
    }
  }

  RETURN_NOT_OK(SplitFixedWidthValueBuffer(rb));
  RETURN_NOT_OK(SplitFixedWidthValidityBuffer(rb));
  RETURN_NOT_OK(SplitBinaryArray(rb));
  RETURN_NOT_OK(SplitLargeBinaryArray(rb));

  // update partition buffer base
  for (auto pid = 0; pid < num_partitions_; ++pid) {
    partition_buffer_base_[pid] += partition_id_cnt_[pid];
  }
  return arrow::Status::OK();
}

arrow::Status Splitter::SplitFixedWidthValueBuffer(const arrow::RecordBatch& rb) {
  const auto num_rows = rb.num_rows();
  for (auto col = 0; col < fixed_width_array_idx_.size(); ++col) {
    std::fill(std::begin(partition_buffer_offset_), std::end(partition_buffer_offset_),
              0);
    auto src_addr = const_cast<uint8_t*>(
        rb.column_data(fixed_width_array_idx_[col])->buffers[1]->data());
    const auto& dst_addrs = partition_fixed_width_value_addrs_[col];
    switch (column_type_id_[fixed_width_array_idx_[col]]) {
#define PROCESS(SHUFFLE_TYPE, CTYPE)                                                     \
  case Type::SHUFFLE_TYPE:                                                               \
    for (auto row = 0; row < num_rows; ++row) {                                          \
      auto pid = partition_id_[row];                                                     \
      reinterpret_cast<CTYPE*>(                                                          \
          dst_addrs[pid])[partition_buffer_base_[pid] + partition_buffer_offset_[pid]] = \
          reinterpret_cast<CTYPE*>(src_addr)[row];                                       \
      partition_buffer_offset_[pid]++;                                                   \
    }                                                                                    \
    break;
      PROCESS(SHUFFLE_1BYTE, uint8_t)
      PROCESS(SHUFFLE_2BYTE, uint16_t)
      PROCESS(SHUFFLE_4BYTE, uint32_t)
      PROCESS(SHUFFLE_8BYTE, uint64_t)
#undef PROCESS
      case Type::SHUFFLE_DECIMAL128:
        for (auto row = 0; row < num_rows; ++row) {
          auto pid = partition_id_[row];
          auto dst_offset = (partition_buffer_base_[pid] + partition_buffer_offset_[pid])
                            << 1;
          reinterpret_cast<uint64_t*>(dst_addrs[pid])[dst_offset] =
              reinterpret_cast<uint64_t*>(src_addr)[row << 1];
          reinterpret_cast<uint64_t*>(dst_addrs[pid])[dst_offset | 1] =
              reinterpret_cast<uint64_t*>(src_addr)[row << 1 | 1];
          partition_buffer_offset_[pid]++;
        }
        break;
      case Type::SHUFFLE_BIT:
        for (auto row = 0; row < num_rows; ++row) {
          auto pid = partition_id_[row];
          auto dst_offset = partition_buffer_base_[pid] + partition_buffer_offset_[pid];
          dst_addrs[pid][dst_offset >> 3] ^=
              (dst_addrs[pid][dst_offset >> 3] >> (dst_offset & 7) ^
               src_addr[row >> 3] >> (row & 7))
              << (dst_offset & 7);
          partition_buffer_offset_[pid]++;
        }
        break;
      default:
        return arrow::Status::Invalid(
            "Column type " +
            schema_->field(fixed_width_array_idx_[col])->type()->ToString() +
            " is not fixed width");
    }
  }
  return arrow::Status::OK();
}

arrow::Status Splitter::SplitFixedWidthValidityBuffer(const arrow::RecordBatch& rb) {
  const auto num_rows = rb.num_rows();
  for (auto col = 0; col < fixed_width_array_idx_.size(); ++col) {
    auto& dst_addrs = partition_fixed_width_validity_addrs_[col];
    if (rb.column_data(fixed_width_array_idx_[col])->GetNullCount() == 0) {
      for (auto pid = 0; pid < num_partitions_; ++pid) {
        if (partition_id_cnt_[pid] > 0 && dst_addrs[pid] != nullptr) {
          arrow::BitUtil::SetBitsTo(dst_addrs[pid], partition_buffer_base_[pid],
                                    partition_id_cnt_[pid], true);
        }
      }
    } else {
      for (auto pid = 0; pid < num_partitions_; ++pid) {
        if (partition_id_cnt_[pid] > 0 && dst_addrs[pid] == nullptr) {
          // init bitmap if it's null
          std::unique_ptr<arrow::TypedBufferBuilder<bool>> null_bitmap_builder =
              std::make_unique<arrow::TypedBufferBuilder<bool>>(options_.memory_pool);
          ARROW_ASSIGN_OR_RAISE(
              auto validity_buffer,
              arrow::AllocateEmptyBitmap(options_.buffer_size, options_.memory_pool));
          dst_addrs[pid] = const_cast<uint8_t*>(validity_buffer->data());
          arrow::BitUtil::SetBitsTo(dst_addrs[pid], 0, partition_buffer_base_[pid], true);
          partition_fixed_width_buffers_[col][pid][0] = std::move(validity_buffer);
        }
      }

      auto src_addr = const_cast<uint8_t*>(
          rb.column_data(fixed_width_array_idx_[col])->buffers[0]->data());
      std::fill(std::begin(partition_buffer_offset_), std::end(partition_buffer_offset_),
                0);
      for (auto row = 0; row < num_rows; ++row) {
        auto pid = partition_id_[row];
        auto dst_offset = partition_buffer_base_[pid] + partition_buffer_offset_[pid];
        dst_addrs[pid][dst_offset >> 3] ^=
            (dst_addrs[pid][dst_offset >> 3] >> (dst_offset & 7) ^
             src_addr[row >> 3] >> (row & 7))
            << (dst_offset & 7);
        partition_buffer_offset_[pid]++;
      }
    }
  }
  return arrow::Status::OK();
}

arrow::Status Splitter::SplitBinaryArray(const arrow::RecordBatch& rb) {
  for (int i = 0; i < binary_array_idx_.size(); ++i) {
    RETURN_NOT_OK(AppendBinary<arrow::BinaryType>(
        std::static_pointer_cast<arrow::BinaryArray>(rb.column(binary_array_idx_[i])),
        partition_binary_builders_[i], rb.num_rows()));
  }
  return arrow::Status::OK();
}

arrow::Status Splitter::SplitLargeBinaryArray(const arrow::RecordBatch& rb) {
  for (int i = 0; i < large_binary_array_idx_.size(); ++i) {
    RETURN_NOT_OK(AppendBinary<arrow::LargeBinaryType>(
        std::static_pointer_cast<arrow::LargeBinaryArray>(
            rb.column(large_binary_array_idx_[i])),
        partition_large_binary_builders_[i], rb.num_rows()));
  }
  return arrow::Status::OK();
}

template <typename T, typename ArrayType, typename BuilderType>
arrow::Status Splitter::AppendBinary(
    const std::shared_ptr<ArrayType>& src_arr,
    const std::vector<std::shared_ptr<BuilderType>>& dst_builders, int64_t num_rows) {
  using offset_type = typename T::offset_type;
  if (src_arr->null_count() == 0) {
    for (auto row = 0; row < num_rows; ++row) {
      offset_type length;
      auto value = src_arr->GetValue(row, &length);
      RETURN_NOT_OK(dst_builders[partition_id_[row]]->Append(value, length));
    }
  } else {
    for (auto row = 0; row < num_rows; ++row) {
      if (src_arr->IsValid(row)) {
        offset_type length;
        auto value = src_arr->GetValue(row, &length);
        RETURN_NOT_OK(dst_builders[partition_id_[row]]->Append(value, length));
      } else {
        RETURN_NOT_OK(dst_builders[partition_id_[row]]->AppendNull());
      }
    }
  }
  return arrow::Status::OK();
}

arrow::Status Splitter::CreatePartitionWriter(int32_t partition_id) {
  ARROW_ASSIGN_OR_RAISE(auto spilled_file_dir,
                        GetSpilledShuffleFileDir(configured_dirs_[dir_selection_],
                                                 sub_dir_selection_[dir_selection_]))
  sub_dir_selection_[dir_selection_] =
      (sub_dir_selection_[dir_selection_] + 1) % options_.num_sub_dirs;
  dir_selection_ = (dir_selection_ + 1) % configured_dirs_.size();
  ARROW_ASSIGN_OR_RAISE(auto spilled_file, CreateTempShuffleFile(spilled_file_dir));
  partition_writer_[partition_id] = std::make_shared<PartitionWriter>(this, std::move(spilled_file));
  return arrow::Status::OK();
}

// ----------------------------------------------------------------------
// RoundRobinSplitter

arrow::Result<std::shared_ptr<RoundRobinSplitter>> RoundRobinSplitter::Create(
    int32_t num_partitions, std::shared_ptr<arrow::Schema> schema, SplitOptions options) {
  std::shared_ptr<RoundRobinSplitter> res(
      new RoundRobinSplitter(num_partitions, std::move(schema), std::move(options)));
  RETURN_NOT_OK(res->Init());
  return res;
}

arrow::Status RoundRobinSplitter::ComputeAndCountPartitionId(
    const arrow::RecordBatch& rb) {
  std::fill(std::begin(partition_id_cnt_), std::end(partition_id_cnt_), 0);
  partition_id_.resize(rb.num_rows());
  for (auto& pid : partition_id_) {
    pid = pid_selection_;
    partition_id_cnt_[pid_selection_]++;
    pid_selection_ = (pid_selection_ + 1) % num_partitions_;
  }
  return arrow::Status::OK();
}

// ----------------------------------------------------------------------
// HashSplitter

arrow::Result<std::shared_ptr<HashSplitter>> HashSplitter::Create(
    int32_t num_partitions, std::shared_ptr<arrow::Schema> schema,
    const gandiva::ExpressionVector& expr_vector, SplitOptions options) {
  std::shared_ptr<HashSplitter> res(
      new HashSplitter(num_partitions, std::move(schema), std::move(options)));
  RETURN_NOT_OK(res->Init());
  RETURN_NOT_OK(res->CreateProjector(expr_vector));
  return res;
}

arrow::Status HashSplitter::CreateProjector(
    const gandiva::ExpressionVector& expr_vector) {
  // same seed as spark's
  auto hash = gandiva::TreeExprBuilder::MakeLiteral((int32_t)42);
  for (const auto& expr : expr_vector) {
    switch (expr->root()->return_type()->id()) {
      case arrow::NullType::type_id:
        break;
      case arrow::BooleanType::type_id:
      case arrow::Int8Type::type_id:
      case arrow::Int16Type::type_id:
      case arrow::Int32Type::type_id:
      case arrow::FloatType::type_id:
      case arrow::Date32Type::type_id:
        hash = gandiva::TreeExprBuilder::MakeFunction(
            "hash32_spark", {expr->root(), hash}, arrow::int32());
        break;
      case arrow::Int64Type::type_id:
      case arrow::DoubleType::type_id:
        hash = gandiva::TreeExprBuilder::MakeFunction(
            "hash64_spark", {expr->root(), hash}, arrow::int32());
        break;
      case arrow::StringType::type_id:
        hash = gandiva::TreeExprBuilder::MakeFunction(
            "hashbuf_spark", {expr->root(), hash}, arrow::int32());
        break;
      default:
        hash = gandiva::TreeExprBuilder::MakeFunction("hash32", {expr->root(), hash},
                                                      arrow::int32());
        /*return arrow::Status::NotImplemented("HashSplitter::CreateProjector doesn't
           support type ", expr->result()->type()->ToString());*/
    }
  }
  auto hash_expr =
      gandiva::TreeExprBuilder::MakeExpression(hash, arrow::field("pid", arrow::int32()));
  return gandiva::Projector::Make(schema_, {hash_expr}, &projector_);
}

arrow::Status HashSplitter::ComputeAndCountPartitionId(const arrow::RecordBatch& rb) {
  auto num_rows = rb.num_rows();
  partition_id_.resize(num_rows);
  std::fill(std::begin(partition_id_cnt_), std::end(partition_id_cnt_), 0);

  arrow::ArrayVector outputs;
  TIME_NANO_OR_RAISE(total_compute_pid_time_,
                     projector_->Evaluate(rb, arrow::default_memory_pool(), &outputs));
  if (outputs.size() != 1) {
    return arrow::Status::Invalid("Projector result should have one field, actual is ",
                                  std::to_string(outputs.size()));
  }
  auto pid_arr = std::dynamic_pointer_cast<arrow::Int32Array>(outputs.at(0));
  for (auto i = 0; i < num_rows; ++i) {
    // positive mod
    auto pid = pid_arr->Value(i) % num_partitions_;
    if (pid < 0) pid = (pid + num_partitions_) % num_partitions_;
    partition_id_[i] = pid;
    partition_id_cnt_[pid]++;
  }
  return arrow::Status::OK();
}

// ----------------------------------------------------------------------
// FallBackRangeSplitter

arrow::Result<std::shared_ptr<FallbackRangeSplitter>> FallbackRangeSplitter::Create(
    int32_t num_partitions, std::shared_ptr<arrow::Schema> schema, SplitOptions options) {
  auto res = std::shared_ptr<FallbackRangeSplitter>(
      new FallbackRangeSplitter(num_partitions, std::move(schema), std::move(options)));
  RETURN_NOT_OK(res->Init());
  return res;
}

arrow::Status FallbackRangeSplitter::Init() {
  input_schema_ = std::move(schema_);
  ARROW_ASSIGN_OR_RAISE(schema_, input_schema_->RemoveField(0))
  return Splitter::Init();
}

arrow::Status FallbackRangeSplitter::Split(const arrow::RecordBatch& rb) {
  EVAL_START("split", options_.thread_id)
  RETURN_NOT_OK(ComputeAndCountPartitionId(rb));
  ARROW_ASSIGN_OR_RAISE(auto remove_pid, rb.RemoveColumn(0));
  RETURN_NOT_OK(DoSplit(*remove_pid));
  EVAL_END("split", options_.thread_id, options_.task_attempt_id)
  return arrow::Status::OK();
}

arrow::Status FallbackRangeSplitter::ComputeAndCountPartitionId(
    const arrow::RecordBatch& rb) {
  if (rb.column(0)->type_id() != arrow::Type::INT32) {
    return arrow::Status::Invalid("RecordBatch field 0 should be ",
                                  arrow::int32()->ToString(), ", actual is ",
                                  rb.column(0)->type()->ToString());
  }

  auto pid_arr = reinterpret_cast<const int32_t*>(rb.column_data(0)->buffers[1]->data());
  auto num_rows = rb.num_rows();
  partition_id_.resize(num_rows);
  std::fill(std::begin(partition_id_cnt_), std::end(partition_id_cnt_), 0);
  for (auto i = 0; i < num_rows; ++i) {
    auto pid = pid_arr[i];
    if (pid >= num_partitions_) {
      return arrow::Status::Invalid("Partition id ", std::to_string(pid),
                                    " is equal or greater than ",
                                    std::to_string(num_partitions_));
    }
    partition_id_[i] = pid;
    partition_id_cnt_[pid]++;
  }
  return arrow::Status::OK();
}

}  // namespace shuffle
}  // namespace sparkcolumnarplugin
