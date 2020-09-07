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

#include <chrono>
#include <memory>

#include <arrow/array.h>
#include <arrow/io/file.h>
#include <arrow/ipc/api.h>
#include <arrow/record_batch.h>
#include <sys/sendfile.h>
#include <iostream>

#include "shuffle/partition_writer.h"
#include "shuffle/utils.h"

namespace sparkcolumnarplugin {
namespace shuffle {

arrow::Result<std::shared_ptr<PartitionWriter>> PartitionWriter::Create(
    int32_t partition_id, int64_t capacity, arrow::Compression::type compression_type,
    Type::typeId last_type, const std::vector<Type::typeId>& column_type_id,
    const std::shared_ptr<arrow::Schema>& schema, int data_file_fd, int spilled_file_fd,
    const std::shared_ptr<arrow::io::FileOutputStream>& data_file_os,
    const std::shared_ptr<arrow::io::FileOutputStream>& spilled_file_os) {
  auto buffers = TypeBufferInfos(Type::NUM_TYPES);
  auto binary_bulders = BinaryBuilders();
  auto large_binary_bulders = LargeBinaryBuilders();

  for (auto type_id : column_type_id) {
    switch (type_id) {
      case Type::SHUFFLE_BINARY: {
        auto builder =
            std::make_unique<arrow::BinaryBuilder>(arrow::default_memory_pool());
        binary_bulders.push_back(std::move(builder));
      } break;
      case Type::SHUFFLE_LARGE_BINARY: {
        auto builder =
            std::make_unique<arrow::LargeBinaryBuilder>(arrow::default_memory_pool());
        large_binary_bulders.push_back(std::move(builder));
      } break;
      case Type::SHUFFLE_NULL: {
        buffers[type_id].push_back(std::make_unique<BufferInfo>(
            BufferInfo{.validity_buffer = nullptr, .value_buffer = nullptr}));
      } break;
      default: {
        std::shared_ptr<arrow::Buffer> validity_buffer;
        std::shared_ptr<arrow::Buffer> value_buffer;
        uint8_t* validity_addr;
        uint8_t* value_addr;

        ARROW_ASSIGN_OR_RAISE(validity_buffer, arrow::AllocateEmptyBitmap(capacity))
        if (type_id == Type::SHUFFLE_BIT) {
          ARROW_ASSIGN_OR_RAISE(value_buffer, arrow::AllocateEmptyBitmap(capacity))
        } else {
          ARROW_ASSIGN_OR_RAISE(value_buffer,
                                arrow::AllocateBuffer(capacity * (1 << type_id)))
        }
        validity_addr = validity_buffer->mutable_data();
        value_addr = value_buffer->mutable_data();
        buffers[type_id].push_back(std::make_unique<BufferInfo>(
            BufferInfo{.validity_buffer = std::move(validity_buffer),
                       .value_buffer = std::move(value_buffer),
                       .validity_addr = validity_addr,
                       .value_addr = value_addr}));
      } break;
    }
  }

  return std::make_shared<PartitionWriter>(
      partition_id, capacity, compression_type, last_type, column_type_id, schema,
      data_file_fd, spilled_file_fd, data_file_os, spilled_file_os, std::move(buffers),
      std::move(binary_bulders), std::move(large_binary_bulders));
}

arrow::Status PartitionWriter::Stop() {
  if (spilled_file_writer_ != nullptr) {
    TIME_NANO_OR_RAISE(spill_time_, Spill(true));
  }
  auto start_write = std::chrono::steady_clock::now();
  ARROW_ASSIGN_OR_RAISE(auto before_write, data_file_os_->Tell());
  if (!spilled_file_offsets_.empty()) {
    for (const auto& offset : spilled_file_offsets_) {
      auto off = offset.first;
      int ret = 0;
      int bytes_written = 0;
      while (ret != -1 && bytes_written < offset.second) {
        ret = sendfile(data_file_fd_, spilled_file_fd_, &off,
                       offset.second - bytes_written);
        if (ret != -1) {
          bytes_written += ret;
        }
      }
      if (ret == -1) {
        return arrow::internal::IOErrorFromErrno(errno, "Error in sendfile");
      }
    }
  } else {
    ARROW_ASSIGN_OR_RAISE(
        auto data_file_writer,
        arrow::ipc::NewStreamWriter(data_file_os_.get(), schema_,
                                    SplitterIpcWriteOptions(compression_type_)));
    ARROW_ASSIGN_OR_RAISE(auto batch, MakeRecordBatchAndReset());
    if (batch == nullptr) {
      return arrow::Status::Invalid("Partition writer can't have emtpy partition");
    }
    RETURN_NOT_OK(data_file_writer->WriteRecordBatch(*batch));
    RETURN_NOT_OK(data_file_writer->Close());
  }

  ARROW_ASSIGN_OR_RAISE(auto after_write, data_file_os_->Tell());
  partition_length_ = after_write - before_write;

  auto end_write = std::chrono::steady_clock::now();
  write_time_ +=
      std::chrono::duration_cast<std::chrono::nanoseconds>(end_write - start_write)
          .count();
  return arrow::Status::OK();
}

arrow::Status PartitionWriter::Spill(bool close_writer) {
  ARROW_ASSIGN_OR_RAISE(auto batch, MakeRecordBatchAndReset());
  if (batch != nullptr) {
    if (spilled_file_writer_ == nullptr) {
      ARROW_ASSIGN_OR_RAISE(
          spilled_file_writer_,
          arrow::ipc::NewStreamWriter(spilled_file_os_.get(), schema_,
                                      SplitterIpcWriteOptions(compression_type_)));
    }
    ARROW_ASSIGN_OR_RAISE(auto before_write, spilled_file_os_->Tell());
    RETURN_NOT_OK(spilled_file_writer_->WriteRecordBatch(*batch));
    if (close_writer) {
      RETURN_NOT_OK(spilled_file_writer_->Close());
    }
    ARROW_ASSIGN_OR_RAISE(auto after_write, spilled_file_os_->Tell());
    spilled_file_offsets_.emplace_back(before_write, after_write - before_write);
  }
  return arrow::Status::OK();
}

arrow::Result<std::shared_ptr<arrow::RecordBatch>>
PartitionWriter::MakeRecordBatchAndReset() {
  if (write_offset_[last_type_] == 0) {
    return nullptr;
  }
  std::vector<std::shared_ptr<arrow::Array>> arrays(schema_->num_fields());
  for (int i = 0; i < schema_->num_fields(); ++i) {
    auto type_id = column_type_id_[i];
    if (type_id == Type::SHUFFLE_BINARY) {
      auto builder = std::move(binary_builders_.front());
      binary_builders_.pop_front();
      RETURN_NOT_OK(builder->Finish(&arrays[i]));
      binary_builders_.push_back(std::move(builder));
    } else if (type_id == Type::SHUFFLE_LARGE_BINARY) {
      auto builder = std::move(large_binary_builders_.front());
      large_binary_builders_.pop_front();
      RETURN_NOT_OK(builder->Finish(&arrays[i]));
      large_binary_builders_.push_back(std::move(builder));
    } else {
      auto buf_info_ptr = std::move(buffers_[type_id].front());
      buffers_[type_id].pop_front();
      auto arr = arrow::ArrayData::Make(
          schema_->field(i)->type(), write_offset_[last_type_],
          std::vector<std::shared_ptr<arrow::Buffer>>{buf_info_ptr->validity_buffer,
                                                      buf_info_ptr->value_buffer});
      arrays[i] = arrow::MakeArray(arr);
      buffers_[type_id].push_back(std::move(buf_info_ptr));
    }
  }
  auto rb = std::move(
      arrow::RecordBatch::Make(schema_, write_offset_[last_type_], std::move(arrays)));
  std::fill(std::begin(write_offset_), std::end(write_offset_), 0);
  return rb;
}

}  // namespace shuffle
}  // namespace sparkcolumnarplugin
