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

class BenchmarkShuffleSplit : public ::testing::TestWithParam<std::tuple<int, int>> {
 public:
  void SetUp() override {
    // read input from parquet file
#ifdef BENCHMARK_FILE_PATH
    std::string dir_path = BENCHMARK_FILE_PATH;
#else
    std::string dir_path = "";
#endif
    std::string path = dir_path + "409600.parquet";
    std::shared_ptr<arrow::fs::FileSystem> fs;
    std::string file_name;
    ARROW_ASSIGN_OR_THROW(fs, arrow::fs::FileSystemFromUriOrPath(path, &file_name))

    ARROW_ASSIGN_OR_THROW(file, fs->OpenInputFile(file_name));

    parquet::ArrowReaderProperties properties(true);
    properties.set_batch_size(4096);

    ASSERT_NOT_OK(::parquet::arrow::FileReader::Make(
        arrow::default_memory_pool(), ::parquet::ParquetFileReader::Open(file),
        properties, &parquet_reader));

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

    ASSERT_NOT_OK(parquet_reader->GetRecordBatchReader(row_group_indices, column_indices,
                                                       &record_batch_reader));

    expr_vector.reserve(num_columns);
    const auto& fields = schema->fields();
    for (const auto& field : fields) {
      auto node = gandiva::TreeExprBuilder::MakeField(field);
      expr_vector.push_back(gandiva::TreeExprBuilder::MakeExpression(
          std::move(node), arrow::field("res_" + field->name(), field->type())));
    }
  }

  void TearDown() override {}

 protected:
  std::shared_ptr<arrow::io::RandomAccessFile> file;
  std::unique_ptr<::parquet::arrow::FileReader> parquet_reader;
  std::shared_ptr<RecordBatchReader> record_batch_reader;
  std::shared_ptr<arrow::Schema> schema;
  std::vector<std::shared_ptr<::gandiva::Expression>> expr_vector;

  std::shared_ptr<Splitter> splitter;

  void doSplit(arrow::Compression::type compression_type) {
    int num_pid = std::get<0>(GetParam());
    int buffer_size = std::get<1>(GetParam());

    auto options = SplitOptions::Defaults();
    options.compression_type = compression_type;
    options.buffer_size = buffer_size;
    ARROW_ASSIGN_OR_THROW(splitter, Splitter::Make("hash", schema, num_pid, expr_vector,
                                                   std::move(options)));

    std::shared_ptr<arrow::RecordBatch> record_batch;
    auto rand = arrow::random::RandomArrayGenerator(0x5487655);
    uint64_t elapse_read = 0;
    uint64_t num_batches = 0;

    auto start = std::chrono::steady_clock::now();
    do {
      TIME_NANO_OR_THROW(elapse_read, record_batch_reader->ReadNext(&record_batch));
      if (record_batch) {
        ASSERT_NOT_OK(splitter->Split(*record_batch));
        num_batches += 1;
      }
    } while (record_batch);

    ASSERT_NOT_OK(splitter->Stop());
    auto end = std::chrono::steady_clock::now();
    auto total_time =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    std::cout << "Setting num_pid to " << num_pid << ", buffer_size to " << buffer_size
              << std::endl;
    std::cout << "Total batches read:  " << num_batches << std::endl;

    std::cout << "Total bytes written: " << splitter->TotalBytesWritten() << std::endl;

    auto compute_pid_time = splitter->TotalComputePidTime();
    auto write_time = splitter->TotalWriteTime();
    auto spill_time = splitter->TotalSpillTime();
    auto split_time =
        total_time - elapse_read - compute_pid_time - spill_time - write_time;
    std::cout << "Took " << TIME_NANO_TO_STRING(elapse_read) << " to read data"
              << std::endl
              << "Took " << TIME_NANO_TO_STRING(compute_pid_time) << " to compute pid"
              << std::endl
              << "Took " << TIME_NANO_TO_STRING(split_time) << " to split" << std::endl
              << "Took " << TIME_NANO_TO_STRING(spill_time) << " to spill" << std::endl
              << "Took " << TIME_NANO_TO_STRING(write_time) << " to write to disk"
              << std::endl;
  }
};

TEST_P(BenchmarkShuffleSplit, LZ4) { doSplit(arrow::Compression::LZ4_FRAME); }

TEST_P(BenchmarkShuffleSplit, Uncompressed) { doSplit(arrow::Compression::UNCOMPRESSED); }

INSTANTIATE_TEST_CASE_P(ShuffleSplit, BenchmarkShuffleSplit,
                        ::testing::Values(std::make_tuple(336, 1 << 12),
                                          std::make_tuple(336, 1 << 13),
                                          std::make_tuple(336, 1 << 14)));

}  // namespace shuffle
}  // namespace sparkcolumnarplugin
