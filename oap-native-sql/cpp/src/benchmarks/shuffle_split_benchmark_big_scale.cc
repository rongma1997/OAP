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

#include <arrow/filesystem/filesystem.h>
#include <arrow/io/interfaces.h>
#include <arrow/memory_pool.h>
#include <arrow/record_batch.h>
#include <arrow/testing/gtest_util.h>
#include <arrow/testing/random.h>
#include <arrow/type.h>
#include <arrow/util/io_util.h>
#include <gtest/gtest.h>
#include <parquet/arrow/reader.h>
#include <parquet/file_reader.h>
#include <shuffle/splitter.h>
#include <chrono>
#include "codegen/code_generator.h"
#include "codegen/code_generator_factory.h"
#include "tests/test_utils.h"

namespace sparkcolumnarplugin {
namespace shuffle {

class BenchmarkShuffleSplitBigScale
    : public ::testing::TestWithParam<std::tuple<int, int>> {
 public:
  void SetUp() override {
    // read input from parquet file
#ifdef BENCHMARK_FILE_PATH
    std::string dir_path = BENCHMARK_FILE_PATH;
#else
    std::string dir_path = "";
#endif
    std::string path = dir_path + "4M.parquet";
    std::shared_ptr<arrow::fs::FileSystem> fs;
    std::string file_name;
    ARROW_ASSIGN_OR_THROW(fs, arrow::fs::FileSystemFromUriOrPath(path, &file_name))

    ARROW_ASSIGN_OR_THROW(file, fs->OpenInputFile(file_name));

    parquet::ArrowReaderProperties properties(true);
    properties.set_batch_size(4096);

    auto pool = arrow::default_memory_pool();
    ASSERT_NOT_OK(::parquet::arrow::FileReader::Make(
        pool, ::parquet::ParquetFileReader::Open(file), properties, &parquet_reader));

    ASSERT_NOT_OK(parquet_reader->GetSchema(&schema));

    auto num_rowgroups = parquet_reader->num_row_groups();
    std::vector<int> row_group_indices;
    for (int i = 0; i < num_rowgroups; ++i) {
      row_group_indices.push_back(i);
    }

    auto num_columns = schema->num_fields();
    std::vector<int> column_indices;
    for (int i = 0; i < num_columns; ++i) {
      column_indices.push_back(i);
    }

    ASSERT_NOT_OK(parquet_reader->GetRecordBatchReader(
        row_group_indices, column_indices, &record_batch_reader));

    std::shared_ptr<arrow::Schema> splitter_schema;
    ARROW_ASSIGN_OR_THROW(splitter_schema,
                          schema->AddField(0, arrow::field("f_pid", arrow::int32())))
    ARROW_ASSIGN_OR_THROW(splitter, Splitter::Make(splitter_schema))
    splitter->set_buffer_size(100);
    splitter->set_compression_codec(arrow::Compression::UNCOMPRESSED);
  }

  void TearDown() override {
    auto& file_infos = splitter->GetPartitionFileInfo();
    std::vector<std::string> file_names;
    file_names.reserve(file_infos.size());
    std::transform(std::begin(file_infos), std::end(file_infos),
                   std::back_inserter(file_names),
                   [](auto& info) { return std::move(info.second); });

    std::for_each(std::cbegin(file_names), std::cend(file_names),
                  [](const auto& file_name) {
                    arrow::internal::DeleteFile(
                        *arrow::internal::PlatformFilename::FromString(file_name));
                  });
  }

 protected:
  std::shared_ptr<arrow::io::RandomAccessFile> file;
  std::unique_ptr<::parquet::arrow::FileReader> parquet_reader;
  std::shared_ptr<RecordBatchReader> record_batch_reader;
  std::shared_ptr<arrow::Schema> schema;
  std::shared_ptr<Splitter> splitter;

  std::vector<std::shared_ptr<::gandiva::Expression>> expr_vector;

  void doSplit() {
    int num_pid = std::get<0>(GetParam());
    int buffer_size = std::get<1>(GetParam());

    splitter->set_buffer_size(buffer_size);

    std::vector<std::shared_ptr<arrow::RecordBatch>> result_batch;
    std::shared_ptr<arrow::RecordBatch> record_batch;
    auto rand = arrow::random::RandomArrayGenerator(0x5487655);
    uint64_t elapse_read = 0;
    uint64_t elapse_eval = 0;
    uint64_t num_batches = 0;

    do {
      TIME_MICRO_OR_THROW(elapse_read, record_batch_reader->ReadNext(&record_batch));
      if (record_batch) {
        auto array = rand.Numeric<arrow::Int32Type>(record_batch->num_rows(), 0, num_pid);
        std::shared_ptr<arrow::RecordBatch> input_batch;
        ARROW_ASSIGN_OR_THROW(
            input_batch,
            record_batch->AddColumn(0, arrow::field("f_pid", arrow::int32()), array));
        TIME_MICRO_OR_THROW(elapse_eval, splitter->Split(*input_batch));
        num_batches += 1;
      }
    } while (record_batch);

    TIME_MICRO_OR_THROW(elapse_eval, splitter->Stop());

    auto elapse_write = splitter->TotalWriteTime();
    auto elapse_split = elapse_eval - elapse_write;
    std::cout << "Setting num_pid to " << num_pid << ", buffer_size to " << buffer_size << std::endl;
    std::cout << "Total batches read:  " << num_batches << std::endl;

    std::cout << "Total bytes written: " << splitter->TotalBytesWritten().ValueOrDie()
              << std::endl;

    std::cout << "Took " << TIME_TO_STRING(elapse_read) << " doing Batch read"
              << std::endl
              << "Took " << TIME_TO_STRING(elapse_split) << " doing Batch split"
              << std::endl
              << "Took " << TIME_TO_STRING(elapse_write) << " doing Batch write"
              << std::endl;
  }
};

TEST_P(BenchmarkShuffleSplitBigScale, LZ4) {
  splitter->set_compression_codec(arrow::Compression::LZ4_FRAME);
  doSplit();
}

TEST_P(BenchmarkShuffleSplitBigScale, Uncompressed) {
  splitter->set_compression_codec(arrow::Compression::UNCOMPRESSED);
  doSplit();
}

INSTANTIATE_TEST_CASE_P(
    ShuffleSplit, BenchmarkShuffleSplitBigScale,
    ::testing::Values(std::make_tuple(336, 1 << 12), std::make_tuple(336, 1 << 13), std::make_tuple(336, 1 << 14)));

}  // namespace shuffle
}  // namespace sparkcolumnarplugin
