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

#include "shuffle/partition_writer.h"
#include "shuffle/partitioning_jni_bridge.h"
#include "shuffle/splitter.h"
#include "shuffle/utils.h"

namespace sparkcolumnarplugin {
namespace shuffle {

class Splitter {
 public:
  ~Splitter();

  void set_buffer_size(int64_t buffer_size);

  virtual arrow::Status Split(const arrow::RecordBatch&);

  /***
   * Stop all writers created by this splitter. If the data buffer managed by the writer
   * is not empty, write to output stream as RecordBatch. Then sort the temporary files by
   * partition id.
   * @return
   */
  virtual arrow::Status Stop();

  virtual std::vector<std::pair<int32_t, std::string>> GetPartitionFileInfo() const;

  virtual arrow::Result<int64_t> TotalBytesWritten();

  virtual int64_t TotalWriteTime();

 protected:
  Splitter() = default;

  class Impl;
  std::unique_ptr<Impl> impl_;
};

class SingleSplitter : public Splitter {
 public:
  static arrow::Result<std::shared_ptr<SingleSplitter>> Make(
      std::shared_ptr<arrow::Schema> schema);

  arrow::Status Split(const arrow::RecordBatch& rb) override;

  arrow::Status Stop() override;

  arrow::Result<int64_t> TotalBytesWritten() override;

  int64_t TotalWriteTime() override;

  std::vector<std::pair<int32_t, std::string>> GetPartitionFileInfo() const override;

 private:
  SingleSplitter(std::shared_ptr<arrow::Schema> schema, std::string file_path,
                 arrow::Compression::type compression_codec);

  std::shared_ptr<arrow::Schema> schema_;
  std::string file_path_;
  arrow::Compression::type compression_codec_;

  bool file_os_opened_ = false;
  std::shared_ptr<arrow::io::FileOutputStream> file_os_;

  bool file_writer_opened_ = false;
  std::shared_ptr<arrow::ipc::RecordBatchWriter> file_writer_;

  int64_t bytes_written_ = 0;
  int64_t total_write_time_ = 0;
};

class RoundRobinSplitter : public Splitter {
 public:
  static arrow::Result<std::shared_ptr<RoundRobinSplitter>> Make(
      int32_t num_partitions, std::shared_ptr<arrow::Schema> schema,
      int32_t buffer_size = kDefaultSplitterBufferSize);

 private:
  RoundRobinSplitter(int32_t num_partitions, std::shared_ptr<arrow::Schema> schema,
                     int32_t buffer_size);

  class RoundRobinImpl;
  std::unique_ptr<RoundRobinImpl> impl_;
};

class BaseProjectionSplitter : public Splitter {
 protected:
  BaseProjectionSplitter() = default;
  class BaseProjectionImpl;
};

class RangeSplitter : public BaseProjectionSplitter {
 public:
  static arrow::Result<std::shared_ptr<RangeSplitter>> Make(
      int32_t num_partitions, std::shared_ptr<arrow::Schema> schema,
      gandiva::ExpressionVector expr_vector,
      gandiva::FieldVector field_vector,
      int32_t buffer_size = kDefaultSplitterBufferSize);

 private:
  RangeSplitter(int32_t num_partitions, std::shared_ptr<arrow::Schema> schema,
                gandiva::ExpressionVector expr_vector,
                gandiva::FieldVector field_vector, int32_t buffer_size);
  class RangeImpl;
  std::unique_ptr<RangeImpl> impl_;
};

}  // namespace shuffle
}  // namespace sparkcolumnarplugin
