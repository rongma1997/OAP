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
#include "shuffle/splitter.h"
#include "shuffle/type.h"
#include "tests/test_utils.h"

namespace sparkcolumnarplugin {
namespace shuffle {

class ShuffleTest : public ::testing::Test {
 protected:
  void SetUp() {
    auto f_pid = field("f_pid", arrow::int32());
    auto f_na = field("f_na", arrow::null());
    auto f_int8 = field("f_int8", arrow::int8());
    auto f_int16 = field("f_int16", arrow::int16());
    auto f_uint64 = field("f_uint64", arrow::uint64());
    auto f_bool = field("f_bool", arrow::boolean());
    auto f_string = field("f_string", arrow::utf8());
    auto f_decimal = field("f_decimal128", arrow::decimal(10, 2));

    std::shared_ptr<arrow::internal::TemporaryDir> tmp_dir1;
    std::shared_ptr<arrow::internal::TemporaryDir> tmp_dir2;
    ARROW_ASSIGN_OR_THROW(tmp_dir1,
                          std::move(arrow::internal::TemporaryDir::Make(tmp_dir_prefix)))
    ARROW_ASSIGN_OR_THROW(tmp_dir2,
                          std::move(arrow::internal::TemporaryDir::Make(tmp_dir_prefix)))
    auto config_dirs = tmp_dir1->path().ToString() + "," + tmp_dir2->path().ToString();

    setenv("NATIVESQL_SPARK_LOCAL_DIRS", config_dirs.c_str(), 1);

    input_schema = arrow::schema(
        {f_pid, f_na, f_int8, f_int16, f_uint64, f_bool, f_string, f_decimal});
    ARROW_ASSIGN_OR_THROW(writer_schema, input_schema->RemoveField(0))

    ARROW_ASSIGN_OR_THROW(splitter, Splitter::Make(input_schema));
  }

  void TearDown() { ASSERT_NOT_OK(splitter->Stop()); }

  std::string tmp_dir_prefix = "columnar-shuffle-test";

  std::string c_pid = "[1, 2, 1, 10]";
  std::vector<std::string> input_data = {c_pid,
                                         "[null, null, null, null]",
                                         "[1, 2, 3, null]",
                                         "[1, -1, null, null]",
                                         "[null, null, null, null]",
                                         "[null, 1, 0, null]",
                                         R"(["alice", "bob", null, null])",
                                         R"(["1.01", "2.01", "3.01", null])"};

  std::vector<std::string> output_data_part0 = {
      "[null, null]",       "[1, 3]",    "[1, null]",
      "[null, null]",       "[null, 0]", R"(["alice", null])",
      R"(["1.01", "3.01"])"};

  std::vector<int> valid_pids = {1, 2, 10};
  std::vector<int> write_once_offsets = {2, 1, 1};

  std::shared_ptr<arrow::Schema> input_schema;
  std::shared_ptr<arrow::Schema> writer_schema;
  std::shared_ptr<Splitter> splitter;
};

TEST_F(ShuffleTest, TestSplitterSchema) { ASSERT_EQ(*input_schema, *splitter->schema()); }

TEST_F(ShuffleTest, TestSplitterTypeId) {
  ASSERT_EQ(splitter->column_type_id(0), Type::SHUFFLE_NULL);
  ASSERT_EQ(splitter->column_type_id(1), Type::SHUFFLE_1BYTE);
  ASSERT_EQ(splitter->column_type_id(2), Type::SHUFFLE_2BYTE);
  ASSERT_EQ(splitter->column_type_id(3), Type::SHUFFLE_8BYTE);
  ASSERT_EQ(splitter->column_type_id(4), Type::SHUFFLE_BIT);
}

TEST_F(ShuffleTest, TestWriterAfterSplit) {
  std::shared_ptr<arrow::RecordBatch> input_batch;
  MakeInputBatch(input_data, input_schema, &input_batch);

  ASSERT_NOT_OK(splitter->Split(*input_batch));

  for(size_t i = 0; i < valid_pids.size(); ++i) {
    auto pid = valid_pids[i];
    ASSERT_NE(splitter->writer(pid), nullptr);
    ASSERT_EQ(splitter->writer(pid)->pid(), pid);
    ASSERT_EQ(splitter->writer(pid)->capacity(), kDefaultSplitterBufferSize);
    ASSERT_EQ(splitter->writer(pid)->last_type(), Type::SHUFFLE_BINARY);
    ASSERT_EQ(splitter->writer(pid)->write_offset(), write_once_offsets[i]);
  }

  ASSERT_EQ(splitter->writer(100), nullptr);
}

TEST_F(ShuffleTest, TestMultipleInput) {
  std::shared_ptr<arrow::RecordBatch> input_batch;
  MakeInputBatch(input_data, input_schema, &input_batch);

  for (int t = 0; t < 3; ++t) {
    ASSERT_NOT_OK(splitter->Split(*input_batch));
    for (size_t i = 0; i < valid_pids.size(); ++i) {
      ASSERT_EQ(splitter->writer(valid_pids[i])->write_offset(), write_once_offsets[i] * (t + 1));
    }
  }
}

TEST_F(ShuffleTest, TestCustomBufferSize) {
  int64_t buffer_size = 2;
  splitter->set_buffer_size(buffer_size);

  std::shared_ptr<arrow::RecordBatch> input_batch;
  MakeInputBatch(input_data, input_schema, &input_batch);

  auto expected_offsets = std::vector<std::vector<int>>{{2, 1, 1}, {2, 2, 2}, {2, 1, 1}};

  for (int t = 0; t < 3; ++t) {
    ASSERT_NOT_OK(splitter->Split(*input_batch));
    for (size_t i = 0; i < valid_pids.size(); ++i) {
      ASSERT_EQ(splitter->writer(valid_pids[i])->write_offset(), expected_offsets[t][i]);
    }
  }
}

TEST_F(ShuffleTest, TestCreateTempFile) {
  std::shared_ptr<arrow::RecordBatch> input_batch;
  MakeInputBatch(input_data, input_schema, &input_batch);

  ASSERT_NOT_OK(splitter->Split(*input_batch));
  ASSERT_EQ(splitter->GetPartitionFileInfo().size(), 3);

  ASSERT_NOT_OK(splitter->Split(*input_batch));
  ASSERT_EQ(splitter->GetPartitionFileInfo().size(), 3);

  MakeInputBatch(
      {"[100]", "[null]", "[null]", "[null]", "[null]", "[null]", "[null]", "[null]"},
      input_schema, &input_batch);

  ASSERT_NOT_OK(splitter->Split(*input_batch));
  ASSERT_EQ(splitter->GetPartitionFileInfo().size(), 4);

  auto file_infos = splitter->GetPartitionFileInfo();
  for (size_t i = 0; i < file_infos.size(); ++i) {
    auto pfn = splitter->GetPartitionFileInfo()[i].second;
    ASSERT_EQ(
        *arrow::internal::FileExists(*arrow::internal::PlatformFilename::FromString(pfn)),
        true);
    ASSERT_NE(pfn.find(tmp_dir_prefix), std::string::npos);
  }
}

TEST_F(ShuffleTest, TestWriterMakeArrowRecordBatch) {
  int64_t buffer_size = 2;
  splitter->set_buffer_size(buffer_size);

  std::shared_ptr<arrow::RecordBatch> input_batch;
  std::shared_ptr<arrow::RecordBatch> output_batch;
  MakeInputBatch(input_data, input_schema, &input_batch);
  MakeInputBatch(output_data_part0, writer_schema, &output_batch);

  ASSERT_NOT_OK(splitter->Split(*input_batch));
  ASSERT_NOT_OK(splitter->Split(*input_batch));
  ASSERT_NOT_OK(splitter->Split(*input_batch));

  ASSERT_NOT_OK(splitter->Stop());

  std::shared_ptr<arrow::io::ReadableFile> file_in;
  std::shared_ptr<arrow::ipc::RecordBatchReader> file_reader;
  ARROW_ASSIGN_OR_THROW(file_in,
                        arrow::io::ReadableFile::Open(splitter->writer(1)->file_path()))

  ARROW_ASSIGN_OR_THROW(file_reader, arrow::ipc::RecordBatchStreamReader::Open(file_in))
  ASSERT_EQ(*file_reader->schema(), *writer_schema);

  int num_rb = 3;
  for (int i = 0; i < num_rb; ++i) {
    std::shared_ptr<arrow::RecordBatch> rb;
    ASSERT_NOT_OK(file_reader->ReadNext(&rb));
    ASSERT_NOT_OK(Equals(*output_batch, *rb));
  }
  ASSERT_NOT_OK(file_in->Close())
}

TEST_F(ShuffleTest, TestCustomCompressionCodec) {
  auto compression_codec = arrow::Compression::LZ4_FRAME;
  splitter->set_compression_codec(compression_codec);

  std::shared_ptr<arrow::RecordBatch> input_batch;
  std::shared_ptr<arrow::RecordBatch> output_batch;
  MakeInputBatch(input_data, input_schema, &input_batch);
  MakeInputBatch(output_data_part0, writer_schema, &output_batch);

  ASSERT_NOT_OK(splitter->Split(*input_batch))
  ASSERT_NOT_OK(splitter->Stop())

  std::shared_ptr<arrow::io::ReadableFile> file_in;
  std::shared_ptr<arrow::ipc::RecordBatchReader> file_reader;
  ARROW_ASSIGN_OR_THROW(file_in,
                        arrow::io::ReadableFile::Open(splitter->writer(1)->file_path()))

  ARROW_ASSIGN_OR_THROW(file_reader, arrow::ipc::RecordBatchStreamReader::Open(file_in))
  ASSERT_EQ(*file_reader->schema(), *writer_schema);

  std::shared_ptr<arrow::RecordBatch> rb;
  ASSERT_NOT_OK(file_reader->ReadNext(&rb));
  ASSERT_NOT_OK(Equals(*rb, *output_batch));

  ASSERT_NOT_OK(file_in->Close())
}

}  // namespace shuffle
}  // namespace sparkcolumnarplugin
