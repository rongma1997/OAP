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

#include <random>

#include <arrow/filesystem/filesystem.h>
#include <arrow/filesystem/localfs.h>
#include <arrow/record_batch.h>
#include <gandiva/arrow.h>
#include <gandiva/gandiva_aliases.h>
#include <gandiva/projector.h>

#include "shuffle/partition_writer.h"
#include "shuffle/partitioning_jni_bridge.h"
#include "shuffle/utils.h"

namespace sparkcolumnarplugin {
namespace shuffle {

class Splitter {
 public:
  ~Splitter();

  static arrow::Result<std::shared_ptr<Splitter>> Make(
      const std::string& short_name, std::shared_ptr<arrow::Schema> schema,
      int num_partitions, int32_t buffer_size = kDefaultSplitterBufferSize,
      arrow::Compression::type compression_type = arrow::Compression::UNCOMPRESSED,
      gandiva::ExpressionVector expr_vector = {}, gandiva::FieldVector field_vector = {});

  const std::shared_ptr<arrow::Schema>& schema() const { return schema_; }

  void set_compression_type(arrow::Compression::type compression_type) {
    compression_type_ = compression_type;
  }

  virtual arrow::Status Split(const arrow::RecordBatch&) = 0;

  /***
   * Stop all writers created by this splitter. If the data buffer managed by the writer
   * is not empty, write to output stream as RecordBatch. Then sort the temporary files by
   * partition id.
   * @return
   */
  virtual arrow::Status Stop() = 0;

  virtual arrow::Result<int64_t> TotalBytesWritten() = 0;

  virtual int64_t TotalWriteTime() = 0;

  virtual const std::vector<std::pair<int32_t, std::string>>& GetPartitionFileInfo()
      const {
    return partition_file_info_;
  }

 protected:
  Splitter() = default;
  Splitter(std::shared_ptr<arrow::Schema> schema,
           arrow::Compression::type compression_type)
      : schema_(std::move(schema)), compression_type_(compression_type) {}

  std::shared_ptr<arrow::Schema> schema_;
  arrow::Compression::type compression_type_;

  std::vector<std::pair<int32_t, std::string>> partition_file_info_;
};

class SingleSplitter : public Splitter {
 public:
  ~SingleSplitter() = default;

  static arrow::Result<std::shared_ptr<SingleSplitter>> Create(
      std::shared_ptr<arrow::Schema> schema,
      arrow::Compression::type compression_type = arrow::Compression::UNCOMPRESSED);

  arrow::Status Split(const arrow::RecordBatch& rb) override;

  arrow::Status Stop() override;

  arrow::Result<int64_t> TotalBytesWritten() override;

  int64_t TotalWriteTime() override;

 private:
  SingleSplitter(std::shared_ptr<arrow::Schema> schema,
                 arrow::Compression::type compression_type, std::string output_file_path);

  const std::string file_path_;

  bool file_os_opened_ = false;
  std::shared_ptr<arrow::io::FileOutputStream> file_os_;

  bool file_writer_opened_ = false;
  std::shared_ptr<arrow::ipc::RecordBatchWriter> file_writer_;

  int64_t bytes_written_ = 0;
  int64_t total_write_time_ = 0;
};

class BasePartitionSplitter : public Splitter {
 public:
  arrow::Status Split(const arrow::RecordBatch&) override;

  arrow::Status Stop() override;

  arrow::Result<int64_t> TotalBytesWritten() override;

  int64_t TotalWriteTime() override;

  void set_buffer_size(int64_t buffer_size) { buffer_size_ = buffer_size; };

 protected:
  BasePartitionSplitter(int32_t num_partitions, std::shared_ptr<arrow::Schema> schema,
                        int32_t buffer_size, arrow::Compression::type compression_type)
      : Splitter(std::move(schema), compression_type),
        buffer_size_(buffer_size),
        num_partitions_(num_partitions) {}

  virtual arrow::Status Init();

  virtual arrow::Result<std::vector<std::shared_ptr<PartitionWriter>>> GetPartitionWriter(
      const arrow::RecordBatch& rb) = 0;

  arrow::Result<std::string> CreateDataFile();

  const int32_t num_partitions_;

  std::vector<std::shared_ptr<PartitionWriter>> partition_writer_;

  // partition writer parameters
  int32_t buffer_size_;
  std::vector<Type::typeId> column_type_id_;
  Type::typeId last_type_id_;

  // configured local dirs for temporary output file
  int32_t dir_selection_ = 0;
  std::vector<std::string> configured_dirs_;
};

class RoundRobinSplitter : public BasePartitionSplitter {
 public:
  static arrow::Result<std::shared_ptr<RoundRobinSplitter>> Create(
      int32_t num_partitions, std::shared_ptr<arrow::Schema> schema,
      int32_t buffer_size = kDefaultSplitterBufferSize,
      arrow::Compression::type compression_type = arrow::Compression::UNCOMPRESSED);

 protected:
  arrow::Result<std::vector<std::shared_ptr<PartitionWriter>>> GetPartitionWriter(
      const arrow::RecordBatch& rb) override;

 private:
  RoundRobinSplitter(int32_t num_partitions, std::shared_ptr<arrow::Schema> schema,
                     int32_t buffer_size, arrow::Compression::type compression_type)
      : BasePartitionSplitter(num_partitions, std::move(schema), buffer_size,
                              compression_type) {}

  int32_t pid_selection_ = 0;
};

class BaseProjectionSplitter : public BasePartitionSplitter {
 protected:
  BaseProjectionSplitter() = default;
  BaseProjectionSplitter(int32_t num_partitions, std::shared_ptr<arrow::Schema> schema,
                         int32_t buffer_size, arrow::Compression::type compression_type,
                         gandiva::ExpressionVector expr_vector,
                         gandiva::FieldVector field_vector)
      : BasePartitionSplitter(num_partitions, std::move(schema), buffer_size,
                              compression_type),
        expr_vector_(expr_vector),
        field_vector_(field_vector) {}

  arrow::Status Init() override {
    RETURN_NOT_OK(BasePartitionSplitter::Init());
    RETURN_NOT_OK(CreateProjector());
  }

  virtual arrow::Status CreateProjector() = 0;

  std::shared_ptr<gandiva::Projector> projector_;
  gandiva::ExpressionVector expr_vector_;
  gandiva::FieldVector field_vector_;
};

class HashSplitter : public BaseProjectionSplitter {
 public:
  static arrow::Result<std::shared_ptr<HashSplitter>> Create(
      int32_t num_partitions, std::shared_ptr<arrow::Schema> schema, int32_t buffer_size,
      arrow::Compression::type compression_type, gandiva::ExpressionVector expr_vector,
      gandiva::FieldVector field_vector);

 private:
  HashSplitter(int32_t num_partitions, std::shared_ptr<arrow::Schema> schema,
               int32_t buffer_size, arrow::Compression::type compression_type,
               gandiva::ExpressionVector expr_vector, gandiva::FieldVector field_vector)
      : BaseProjectionSplitter(num_partitions, std::move(schema), buffer_size,
                               compression_type, std::move(expr_vector),
                               std::move(field_vector)) {}

  arrow::Status CreateProjector() override;

  arrow::Result<std::vector<std::shared_ptr<PartitionWriter>>> GetPartitionWriter(
      const arrow::RecordBatch& rb) override;
};

}  // namespace shuffle
}  // namespace sparkcolumnarplugin
