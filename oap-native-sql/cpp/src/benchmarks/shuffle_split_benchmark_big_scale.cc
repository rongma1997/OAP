#include <arrow/filesystem/filesystem.h>
#include <arrow/io/interfaces.h>
#include <arrow/memory_pool.h>
#include <arrow/record_batch.h>
#include <arrow/testing/gtest_util.h>
#include <arrow/testing/random.h>
#include <arrow/type.h>
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

class BenchmarkShuffleSplitBigScale : public ::testing::Test {
 public:
  void SetUp() override {
    // read input from parquet file
#ifdef BENCHMARK_FILE_PATH
    std::string dir_path = BENCHMARK_FILE_PATH;
#else
    std::string dir_path = "";
#endif
    std::string path = dir_path + "409600.parquet";
    std::cout << "This Benchmark used file " << path
              << ", please download from server "
                 "vsr200://home/zhouyuan/sparkColumnarPlugin/source_files"
              << std::endl;
    std::shared_ptr<arrow::fs::FileSystem> fs;
    std::string file_name;
    ARROW_ASSIGN_OR_THROW(fs, arrow::fs::FileSystemFromUri(path, &file_name))

    ARROW_ASSIGN_OR_THROW(file, fs->OpenInputFile(file_name));

    parquet::ArrowReaderProperties properties(true);
    properties.set_batch_size(4096);

    auto pool = arrow::default_memory_pool();
    ASSERT_NOT_OK(::parquet::arrow::FileReader::Make(
        pool, ::parquet::ParquetFileReader::Open(file), properties, &parquet_reader));

    //    ASSERT_NOT_OK(
    //        parquet_reader->GetRecordBatchReader({0}, {0, 1, 2}, &record_batch_reader));
    ASSERT_NOT_OK(parquet_reader->GetRecordBatchReader(
        {0, 1}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13}, &record_batch_reader));

    schema = record_batch_reader->schema();

    ////////////////// expr prepration ////////////////
    field_list = schema->fields();
    std::shared_ptr<arrow::Schema> splitter_schema;
    ARROW_ASSIGN_OR_THROW(splitter_schema,
                          schema->AddField(0, arrow::field("f_pid", arrow::int32())))
    ARROW_ASSIGN_OR_THROW(splitter, Splitter::Make(splitter_schema))

    splitter->set_buffer_size(100);
	num_partitions = 10;
  }

 protected:
  std::shared_ptr<arrow::io::RandomAccessFile> file;
  std::unique_ptr<::parquet::arrow::FileReader> parquet_reader;
  std::shared_ptr<RecordBatchReader> record_batch_reader;
  std::shared_ptr<arrow::Schema> schema;
  std::shared_ptr<Splitter> splitter;

  std::vector<std::shared_ptr<::arrow::Field>> field_list;
  std::vector<std::shared_ptr<::gandiva::Expression>> expr_vector;
  std::vector<std::shared_ptr<::arrow::Field>> ret_field_list;
  int num_partitions;

  void doSplit() {
    std::vector<std::shared_ptr<arrow::RecordBatch>> result_batch;
    std::shared_ptr<arrow::RecordBatch> record_batch;
    auto rand = arrow::random::RandomArrayGenerator(0x5487655);
    uint64_t elapse_read = 0;
    uint64_t elapse_eval = 0;
    uint64_t num_batches = 0;

    do {
      TIME_MICRO_OR_THROW(elapse_read, record_batch_reader->ReadNext(&record_batch));
      if (record_batch) {
        auto array = rand.Numeric<arrow::Int32Type>(record_batch->num_rows(), 0, num_partitions);
        std::shared_ptr<arrow::RecordBatch> input_batch;
        ARROW_ASSIGN_OR_THROW(
            input_batch,
            record_batch->AddColumn(0, arrow::field("f_pid", arrow::int32()), array));
        TIME_MICRO_OR_THROW(elapse_eval, splitter->Split(*input_batch));
        num_batches += 1;
      }
    } while (record_batch);

    TIME_MICRO_OR_THROW(elapse_eval, splitter->Stop());
    std::cout << "Readed " << num_batches << " batches." << std::endl;

    std::cout << "BenchmarkExtractBigScale processed " << num_batches << " batches, took "
              << TIME_TO_STRING(elapse_read) << " doing BatchRead, took "
              << TIME_TO_STRING(elapse_eval) << " doing Batch Evaluation." << std::endl;

    std::cout << "Total bytes written: " << splitter->TotalBytesWritten().ValueOrDie()
              << std::endl;
  }
};

TEST_F(BenchmarkShuffleSplitBigScale, Uncompressed) {
  splitter->set_compression_codec(arrow::Compression::UNCOMPRESSED);
  doSplit();
}

TEST_F(BenchmarkShuffleSplitBigScale, LZ4) {
  splitter->set_compression_codec(arrow::Compression::LZ4_FRAME);
  doSplit();
}

}  // namespace shuffle
}  // namespace sparkcolumnarplugin
