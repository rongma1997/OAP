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

#pragma once

#include <chrono>
#include <vector>

#include <arrow/array/builder_binary.h>
#include <arrow/buffer.h>
#include <arrow/io/file.h>
#include <arrow/ipc/writer.h>
#include <arrow/status.h>
#include <arrow/util/compression.h>

#include "shuffle/type.h"
#include "utils/macros.h"

namespace sparkcolumnarplugin {
namespace shuffle {

namespace detail {

template <typename T>
arrow::Status inline Write(const SrcBuffers& src, int64_t src_offset,
                           const BufferInfos& dst, int64_t dst_offset) {
  for (size_t i = 0; i < src.size(); ++i) {
    if (dst[i]->validity_addr[dst_offset / 8] >> dst_offset % 8 & 1 ^
        (src[i].validity_addr)[src_offset / 8] >> src_offset % 8 & 1) {
      dst[i]->validity_addr[dst_offset / 8] ^= 1 << dst_offset % 8;
    }
    reinterpret_cast<T*>(dst[i]->value_addr)[dst_offset] =
        reinterpret_cast<T*>(src[i].value_addr)[src_offset];
  }
  return arrow::Status::OK();
}

template <>
arrow::Status inline Write<bool>(const SrcBuffers& src, int64_t src_offset,
                                 const BufferInfos& dst, int64_t dst_offset) {
  for (size_t i = 0; i < src.size(); ++i) {
    if (dst[i]->validity_addr[dst_offset / 8] >> dst_offset % 8 & 1 ^
        (src[i].validity_addr)[src_offset / 8] >> src_offset % 8 & 1) {
      dst[i]->validity_addr[dst_offset / 8] ^= 1 << dst_offset % 8;
    }
    if (dst[i]->value_addr[dst_offset / 8] >> dst_offset % 8 & 1 ^
        (src[i].value_addr)[src_offset / 8] >> src_offset % 8 & 1) {
      dst[i]->value_addr[dst_offset / 8] ^= 1 << dst_offset % 8;
    }
  }
  return arrow::Status::OK();
}

arrow::Status inline WriteDecimal128(const SrcBuffers& src, int64_t src_offset,
                                     const BufferInfos& dst, int64_t dst_offset) {
  for (size_t i = 0; i < src.size(); ++i) {
    if (dst[i]->validity_addr[dst_offset / 8] >> dst_offset % 8 & 1 ^
        (src[i].validity_addr)[src_offset / 8] >> src_offset % 8 & 1) {
      dst[i]->validity_addr[dst_offset / 8] ^= 1 << dst_offset % 8;
    }
    reinterpret_cast<uint64_t*>(dst[i]->value_addr)[dst_offset << 1] =
        reinterpret_cast<uint64_t*>(src[i].value_addr)[src_offset << 1];
    reinterpret_cast<int64_t*>(dst[i]->value_addr)[dst_offset << 1 | 1] =
        reinterpret_cast<int64_t*>(src[i].value_addr)[src_offset << 1 | 1];
  }
  return arrow::Status::OK();
}

template <typename T, typename ArrayType = typename arrow::TypeTraits<T>::ArrayType,
          typename BuilderType = typename arrow::TypeTraits<T>::BuilderType>
arrow::enable_if_binary_like<T, arrow::Status> inline WriteBinary(
    const std::vector<std::shared_ptr<ArrayType>>& src, int64_t offset,
    const std::deque<std::unique_ptr<BuilderType>>& builders) {
  using offset_type = typename T::offset_type;

  for (size_t i = 0; i < src.size(); ++i) {
    // check not null
    if (src[i]->IsValid(offset)) {
      offset_type length;
      auto value = src[i]->GetValue(offset, &length);
      RETURN_NOT_OK(builders[i]->Append(value, length));
    } else {
      RETURN_NOT_OK(builders[i]->AppendNull());
    }
  }
  return arrow::Status::OK();
}

}  // namespace detail
class PartitionWriter {
 public:
  PartitionWriter(int32_t partition_id, int64_t capacity,
                  arrow::Compression::type compression_type, Type::typeId last_type,
                  const std::vector<Type::typeId>& column_type_id,
                  const std::shared_ptr<arrow::Schema>& schema,
                  const std::shared_ptr<arrow::io::OutputStream>& data_file_os,
                  std::string spilled_file, TypeBufferInfos buffers,
                  BinaryBuilders binary_builders,
                  LargeBinaryBuilders large_binary_builders)
      : partition_id_(partition_id),
        capacity_(capacity),
        compression_type_(compression_type),
        last_type_(last_type),
        column_type_id_(column_type_id),
        schema_(schema),
        data_file_os_(data_file_os),
        spilled_file_(std::move(spilled_file)),
        buffers_(std::move(buffers)),
        binary_builders_(std::move(binary_builders)),
        large_binary_builders_(std::move(large_binary_builders)),
        write_offset_(Type::typeId::NUM_TYPES) {}

  static arrow::Result<std::shared_ptr<PartitionWriter>> Create(
      int32_t partition_id, int64_t capacity, arrow::Compression::type compression_type,
      Type::typeId last_type, const std::vector<Type::typeId>& column_type_id,
      const std::shared_ptr<arrow::Schema>& schema,
      const std::shared_ptr<arrow::io::OutputStream>& data_file_os,
      std::string spilled_file);

  arrow::Status Stop();

  int64_t GetWriteTime() const { return write_time_; }

  int64_t GetSpillTime() const { return spill_time_; }

  int64_t GetPartitionLength() const { return partition_length_; }

  arrow::Result<bool> inline CheckTypeWriteEnds(const Type::typeId& type_id) {
    if (write_offset_[type_id] == capacity_) {
      if (type_id == last_type_) {
        // Write to spilled file, close the file but don't call RecordBatchWriter.Close()
        // since it may not be the last batch to write
        TIME_NANO_OR_RAISE(spill_time_, Spill());
      }
      return true;
    }
    return false;
  }

  /// Do memory copy, return true if mem-copy performed
  /// if writer's memory buffer is full, then no mem-copy will be performed, will spill to
  /// disk and return false
  /// \tparam T arrow::DataType
  /// \param type_id shuffle type id mapped from T
  /// \param src source buffers
  /// \param offset index of the element in source buffers
  /// \return true if write performed, else false
  template <typename T>
  arrow::Result<bool> inline Write(Type::typeId type_id, const SrcBuffers& src,
                                   int64_t offset) {
    // for the type_id, check if write ends. For the last type reset write_offset and
    // spill
    ARROW_ASSIGN_OR_RAISE(auto write_ends, CheckTypeWriteEnds(type_id))
    if (write_ends) {
      return false;
    }

    RETURN_NOT_OK(
        detail::Write<T>(src, offset, buffers_[type_id], write_offset_[type_id]));

    ++write_offset_[type_id];
    return true;
  }

  /// Do memory copy for binary type
  /// \param src source binary array
  /// \param offset index of the element in source binary array
  /// \return true if write performed, else false
  arrow::Result<bool> inline WriteBinary(
      const std::vector<std::shared_ptr<arrow::BinaryArray>>& src, int64_t offset) {
    ARROW_ASSIGN_OR_RAISE(auto write_ends, CheckTypeWriteEnds(Type::SHUFFLE_BINARY))
    if (write_ends) {
      return false;
    }

    RETURN_NOT_OK(detail::WriteBinary<arrow::BinaryType>(src, offset, binary_builders_));

    ++write_offset_[Type::SHUFFLE_BINARY];
    return true;
  }

  /// Do memory copy for large binary type
  /// \param src source binary array
  /// \param offset index of the element in source binary array
  /// \return
  arrow::Result<bool> inline WriteLargeBinary(
      const std::vector<std::shared_ptr<arrow::LargeBinaryArray>>& src, int64_t offset) {
    ARROW_ASSIGN_OR_RAISE(auto write_ends, CheckTypeWriteEnds(Type::SHUFFLE_LARGE_BINARY))
    if (write_ends) {
      return false;
    }

    RETURN_NOT_OK(
        detail::WriteBinary<arrow::LargeBinaryType>(src, offset, large_binary_builders_));

    ++write_offset_[Type::SHUFFLE_LARGE_BINARY];
    return true;
  }

  /// Do memory copy for decimal
  /// \param src source buffers
  /// \param offset index of the element in source buffers
  /// \return true if write performed, else false
  arrow::Result<bool> inline WriteDecimal128(const SrcBuffers& src, int64_t offset) {
    // for the type_id, check if write ends. For the last type reset write_offset and
    // spill
    ARROW_ASSIGN_OR_RAISE(auto write_ends, CheckTypeWriteEnds(Type::SHUFFLE_DECIMAL128))
    if (write_ends) {
      return false;
    }

    RETURN_NOT_OK(detail::WriteDecimal128(src, offset, buffers_[Type::SHUFFLE_DECIMAL128],
                                          write_offset_[Type::SHUFFLE_DECIMAL128]));

    ++write_offset_[Type::SHUFFLE_DECIMAL128];
    return true;
  }

 private:
  arrow::Status Spill();

  arrow::Result<std::shared_ptr<arrow::RecordBatch>> MakeRecordBatchAndReset();

  const int32_t partition_id_;
  const int64_t capacity_;
  const Type::typeId last_type_;

  // hold references to splitter
  const std::vector<Type::typeId>& column_type_id_;
  const std::shared_ptr<arrow::Schema>& schema_;
  const std::shared_ptr<arrow::io::OutputStream>& data_file_os_;

  std::string spilled_file_;
  std::shared_ptr<arrow::io::OutputStream> spilled_file_os_;
  std::shared_ptr<arrow::ipc::RecordBatchWriter> spilled_file_writer_;

  TypeBufferInfos buffers_;
  BinaryBuilders binary_builders_;
  LargeBinaryBuilders large_binary_builders_;

  arrow::Compression::type compression_type_;

  std::vector<int64_t> write_offset_;

  int64_t write_time_ = 0;
  int64_t spill_time_ = 0;
  int64_t partition_length_ = 0;
};

}  // namespace shuffle
}  // namespace sparkcolumnarplugin
