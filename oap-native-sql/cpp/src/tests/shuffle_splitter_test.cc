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

#include <arrow/io/api.h>
#include <arrow/ipc/message.h>
#include <arrow/ipc/reader.h>
#include <arrow/ipc/util.h>
#include <arrow/record_batch.h>
#include <arrow/util/io_util.h>
#include <gtest/gtest.h>
#include <iostream>
#include "shuffle/partition_splitter.h"
#include "tests/test_utils.h"

namespace sparkcolumnarplugin {
namespace shuffle {

class SplitterTest : public ::testing::Test {
 protected:
  void SetUp() {
    auto f_na = field("f_na", arrow::null());
    auto f_int8 = field("f_int8", arrow::int8());
    auto f_int16 = field("f_int16", arrow::int16());
    auto f_uint64 = field("f_uint64", arrow::uint64());
    auto f_bool = field("f_bool", arrow::boolean());
    auto f_string = field("f_string", arrow::utf8());

    std::shared_ptr<arrow::internal::TemporaryDir> tmp_dir1;
    std::shared_ptr<arrow::internal::TemporaryDir> tmp_dir2;
    ARROW_ASSIGN_OR_THROW(tmp_dir1,
                          std::move(arrow::internal::TemporaryDir::Make(tmp_dir_prefix)))
    ARROW_ASSIGN_OR_THROW(tmp_dir2,
                          std::move(arrow::internal::TemporaryDir::Make(tmp_dir_prefix)))
    auto config_dirs = tmp_dir1->path().ToString() + "," + tmp_dir2->path().ToString();

    setenv("NATIVESQL_SPARK_LOCAL_DIRS", config_dirs.c_str(), 1);

    schema_ = arrow::schema({f_na, f_int8, f_int16, f_uint64, f_bool, f_string});
  }

  static const std::string tmp_dir_prefix;
  static const std::vector<std::string> input_data;

  std::shared_ptr<arrow::Schema> schema_;
  std::shared_ptr<Splitter> splitter_;
};

const std::string SplitterTest::tmp_dir_prefix = "columnar-shuffle-test";
const std::vector<std::string> SplitterTest::input_data = {
    "[null, null, null, null]", "[1, 2, 3, null]",    "[1, -1, null, null]",
    "[null, null, null, null]", "[null, 1, 0, null]", R"(["alice", "bob", null, null])"};

TEST_F(SplitterTest, TestSingleSplitter) {
  ARROW_ASSIGN_OR_THROW(
      splitter_, Splitter::Make("single", schema_, 0, 0, Compression::SNAPPY,
                                gandiva::ExpressionVector(), gandiva::FieldVector()))

  std::shared_ptr<arrow::RecordBatch> input_batch;
  MakeInputBatch(input_data, schema_, &input_batch);

  int split_times = 3;
  for (int i = 0; i < split_times; ++i) {
    ASSERT_NOT_OK(splitter_->Split(*input_batch))
  }

  ASSERT_NOT_OK(splitter_->Stop());

  auto file_info = splitter_->GetPartitionFileInfo();
  ASSERT_EQ(file_info.size(), 1);

  auto file_name = file_info.front().second;
  ASSERT_EQ(*arrow::internal::FileExists(
                *arrow::internal::PlatformFilename::FromString(file_name)),
            true);
  ASSERT_NE(file_name.find(tmp_dir_prefix), std::string::npos);

  std::shared_ptr<arrow::io::ReadableFile> file_in;
  std::shared_ptr<arrow::ipc::RecordBatchReader> file_reader;
  ARROW_ASSIGN_OR_THROW(file_in, arrow::io::ReadableFile::Open(file_name))

  ARROW_ASSIGN_OR_THROW(file_reader, arrow::ipc::RecordBatchStreamReader::Open(file_in))
  ASSERT_EQ(*file_reader->schema(), *splitter_->schema());

  std::shared_ptr<arrow::RecordBatch> rb;
  for (int i = 0; i < split_times; ++i) {
    ASSERT_NOT_OK(file_reader->ReadNext(&rb));
    ASSERT_NOT_OK(Equals(*rb, *input_batch));
  }

  if (!file_in->closed()) {
    ASSERT_NOT_OK(file_in->Close());
  }
}

TEST_F(SplitterTest, TestRoundRobinSplitter) {
  int32_t num_partitions = 3;
  int32_t buffer_size = 3;
  ARROW_ASSIGN_OR_THROW(splitter_,
                        Splitter::Make("rr", schema_, num_partitions, buffer_size))

  std::shared_ptr<arrow::RecordBatch> input_batch;
  MakeInputBatch(input_data, schema_, &input_batch);

  int split_times = 3;
  for (int i = 0; i < split_times; ++i) {
    ASSERT_NOT_OK(splitter_->Split(*input_batch))
  }
  ASSERT_NOT_OK(splitter_->Stop());

  auto file_info = splitter_->GetPartitionFileInfo();

  for (auto & info : file_info) {
    auto file_name = info.second;
    ASSERT_EQ(*arrow::internal::FileExists(
                  *arrow::internal::PlatformFilename::FromString(file_name)),
              true);
    ASSERT_NE(file_name.find(tmp_dir_prefix), std::string::npos);

    std::shared_ptr<arrow::io::ReadableFile> file_in;
    std::shared_ptr<arrow::ipc::RecordBatchReader> file_reader;
    ARROW_ASSIGN_OR_THROW(file_in, arrow::io::ReadableFile::Open(file_name))

    ARROW_ASSIGN_OR_THROW(file_reader, arrow::ipc::RecordBatchStreamReader::Open(file_in))
    ASSERT_EQ(*file_reader->schema(), *splitter_->schema());

    std::shared_ptr<arrow::RecordBatch> rb;
    ASSERT_NOT_OK(file_reader->ReadNext(&rb));
    ASSERT_EQ(rb->num_rows(), buffer_size);

    if (!file_in->closed()) {
      ASSERT_NOT_OK(file_in->Close());
    }
  }
}

TEST_F(SplitterTest, TestHashSplitter) {
  int32_t num_partitions = 3;
  int32_t buffer_size = 3;
  ARROW_ASSIGN_OR_THROW(
      splitter_, Splitter::Make("hash", schema_, num_partitions, buffer_size,
                                arrow::Compression::UNCOMPRESSED, {}, {schema_->fields()}))

  std::shared_ptr<arrow::RecordBatch> input_batch;
  MakeInputBatch(input_data, schema_, &input_batch);

  int split_times = 3;
  for (int i = 0; i < split_times; ++i) {
    ASSERT_NOT_OK(splitter_->Split(*input_batch))
  }
  ASSERT_NOT_OK(splitter_->Stop());

  auto file_info = splitter_->GetPartitionFileInfo();

  for (auto & info : file_info) {
    auto file_name = info.second;
    ASSERT_EQ(*arrow::internal::FileExists(
        *arrow::internal::PlatformFilename::FromString(file_name)),
              true);
    ASSERT_NE(file_name.find(tmp_dir_prefix), std::string::npos);

    std::shared_ptr<arrow::io::ReadableFile> file_in;
    std::shared_ptr<arrow::ipc::RecordBatchReader> file_reader;
    ARROW_ASSIGN_OR_THROW(file_in, arrow::io::ReadableFile::Open(file_name))

    ARROW_ASSIGN_OR_THROW(file_reader, arrow::ipc::RecordBatchStreamReader::Open(file_in))
    ASSERT_EQ(*file_reader->schema(), *splitter_->schema());

    std::shared_ptr<arrow::RecordBatch> rb;
    ASSERT_NOT_OK(file_reader->ReadNext(&rb));
    ASSERT_EQ(rb->num_rows(), buffer_size);

    if (!file_in->closed()) {
      ASSERT_NOT_OK(file_in->Close());
    }
  }
}

}  // namespace shuffle
}  // namespace sparkcolumnarplugin
