#include "shuffle/splitter.h"
#include <arrow/buffer_builder.h>
#include <arrow/filesystem/filesystem.h>
#include <arrow/filesystem/localfs.h>
#include <arrow/filesystem/path_util.h>
#include <arrow/type.h>
#include <arrow/util/io_util.h>
#include "shuffle/partition_writer.h"

#include <algorithm>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>
#include <utility>

namespace sparkcolumnarplugin {
namespace shuffle {

std::string GenerateUUID() {
  boost::uuids::random_generator generator;
  return boost::uuids::to_string(generator());
}

class Splitter::Impl {
 public:
  explicit Impl(const std::shared_ptr<arrow::Schema>& schema) : schema_(schema) {}

  arrow::Status Init() {
    // remove partition id field since we don't need it while splitting
    RETURN_NOT_OK(schema_->RemoveField(0, &writer_schema_));

    const auto& fields = writer_schema_->fields();
    std::vector<Type::typeId> result;
    result.reserve(fields.size());

    std::transform(std::cbegin(fields), std::cend(fields), std::back_inserter(result),
                   [](const std::shared_ptr<arrow::Field>& field) -> Type::typeId {
                     auto arrow_type_id = field->type()->id();
                     switch (arrow_type_id) {
                       case arrow::BooleanType::type_id:
                         return Type::SHUFFLE_BIT;
                       case arrow::Int8Type::type_id:
                       case arrow::UInt8Type::type_id:
                         return Type::SHUFFLE_1BYTE;
                       case arrow::Int16Type::type_id:
                       case arrow::UInt16Type::type_id:
                       case arrow::HalfFloatType::type_id:
                         return Type::SHUFFLE_2BYTE;
                       case arrow::Int32Type::type_id:
                       case arrow::UInt32Type::type_id:
                       case arrow::FloatType::type_id:
                         return Type::SHUFFLE_4BYTE;
                       case arrow::Int64Type::type_id:
                       case arrow::UInt64Type::type_id:
                       case arrow::DoubleType::type_id:
                         return Type::SHUFFLE_8BYTE;
                       case arrow::BinaryType::type_id:
                       case arrow::StringType::type_id:
                         return Type::SHUFFLE_BINARY;
                       case arrow::LargeBinaryType::type_id:
                       case arrow::LargeStringType::type_id:
                         return Type::SHUFFLE_LARGE_BINARY;
                       case arrow::NullType::type_id:
                         return Type::SHUFFLE_NULL;
                       default:
                         std::cout << field->ToString() << " field type id " << arrow_type_id << std::endl;
                         return Type::SHUFFLE_NOT_IMPLEMENTED;
                     }
                   });

    auto it =
        std::find(std::begin(result), std::end(result), Type::SHUFFLE_NOT_IMPLEMENTED);
    if (it != std::end(result)) {
      RETURN_NOT_OK(arrow::Status::NotImplemented("field contains not implemented type"));
    }
    column_type_id_ = std::move(result);

    decltype(column_type_id_) remove_null_id(column_type_id_.size());
    std::copy_if(std::cbegin(column_type_id_), std::cend(column_type_id_),
                 std::begin(remove_null_id),
                 [](Type::typeId id) { return id != Type::typeId::SHUFFLE_NULL; });
    last_type_ =
        *std::max_element(std::cbegin(remove_null_id), std::cend(remove_null_id));

    auto local_fs = std::make_shared<arrow::fs::LocalFileSystem>();
    fs_base_dir_ = *arrow::internal::TemporaryDir::Make("native-shuffle-");
    fs_ = std::make_unique<arrow::fs::SubTreeFileSystem>(fs_base_dir_->path().ToString(),
                                                         local_fs);
    return arrow::Status::OK();
  }

  arrow::Status Split(const arrow::RecordBatch& record_batch) {
    const auto& pid_arr = record_batch.column_data(0);
    if (pid_arr->GetNullCount() != 0) {
      return arrow::Status::Invalid("Column partition id should not contain NULL value");
    }
    if (pid_arr->type->id() != arrow::Int32Type::type_id) {
      return arrow::Status::Invalid("Partition id data type mismatch, expected ",
                                    arrow::Int32Type::type_name(), ", but got ",
                                    record_batch.column(0)->type()->name());
    }

    auto num_rows = record_batch.num_rows();
    auto num_cols = record_batch.num_columns();
    auto src_addr = std::vector<SrcBuffers>(Type::NUM_TYPES);
    auto src_binary_arr = SrcBinaryArrays();
    auto src_large_binary_arr = SrcLargeBinaryArrays();

    // TODO: make dummy_buf private static if possible
    arrow::TypedBufferBuilder<bool> null_bitmap_builder_;
    RETURN_NOT_OK(null_bitmap_builder_.Append(num_rows, true));

    std::shared_ptr<arrow::Buffer> dummy_buf;
    RETURN_NOT_OK(null_bitmap_builder_.Finish(&dummy_buf));
    auto dummy_buf_p = const_cast<uint8_t*>(dummy_buf->data());

    // Get the pointer of each buffer, Ignore column_data(0) which indicates the partition
    // id
    for (auto i = 0; i < num_cols - 1; ++i) {
      const auto& buffers = record_batch.column_data(i + 1)->buffers;
      if (column_type_id_[i] == Type::SHUFFLE_BINARY) {
        src_binary_arr.push_back(
            std::static_pointer_cast<arrow::BinaryArray>(record_batch.column(i + 1)));
      } else if (column_type_id_[i] == Type::SHUFFLE_LARGE_BINARY) {
        src_large_binary_arr.push_back(
            std::static_pointer_cast<arrow::LargeBinaryArray>(record_batch.column(i + 1)));
      } else if (column_type_id_[i] != Type::SHUFFLE_NULL) {
        if (record_batch.column_data(i + 1)->GetNullCount() == 0) {
          // null bitmap may be nullptr
          src_addr[column_type_id_[i]].push_back(
              {.validity_addr = dummy_buf_p,
               .value_addr = const_cast<uint8_t*>(buffers[1]->data())});
        } else {
          src_addr[column_type_id_[i]].push_back(
              {.validity_addr = const_cast<uint8_t*>(buffers[0]->data()),
               .value_addr = const_cast<uint8_t*>(buffers[1]->data())});
        }
      }
    }

    // map discrete partition id (pid) to continuous integer (new_id)
    // create a new writer every time a new_id occurs
    std::vector<int> new_id;
    new_id.reserve(num_rows);
    auto pid_cast_p = reinterpret_cast<const int32_t*>(pid_arr->buffers[1]->data());
    for (int64_t i = 0; i < num_rows; ++i) {
      auto pid = pid_cast_p[i];
      if (pid_to_new_id_.find(pid) == pid_to_new_id_.end()) {
        auto temp_dir = GenerateUUID();
        while ((*fs_->GetFileInfo(temp_dir)).type() !=
               arrow::fs::FileType::NotFound) {
          temp_dir = GenerateUUID();
        }
        RETURN_NOT_OK(fs_->CreateDir(temp_dir));
        auto temp_file_path = arrow::fs::internal::ConcatAbstractPath(
                                  fs_base_dir_->path().ToString(),
                                  (*fs_->GetFileInfo(temp_dir)).path()) +
                              "/data";
        temp_files.push_back({pid, temp_file_path});

        ARROW_ASSIGN_OR_RAISE(auto writer, PartitionWriter::Create(pid, buffer_size_, last_type_,
                                              column_type_id_, writer_schema_,
                                              temp_file_path, compression_codec_));
        pid_writer_.push_back(std::move(writer));
        new_id.push_back(num_partitiions_);
        pid_to_new_id_[pid] = num_partitiions_++;
      } else {
        new_id.push_back(pid_to_new_id_[pid]);
      }
    }

    auto read_offset = 0;

#define WRITE_FIXEDWIDTH(TYPE_ID, T)                                                 \
  if (!src_addr[TYPE_ID].empty()) {                                                  \
    for (i = read_offset; i < num_rows; ++i) {                                       \
      auto result = pid_writer_[new_id[i]]->Write<T>(TYPE_ID, src_addr[TYPE_ID], i); \
      RETURN_NOT_OK(result.status());                                                \
      if (!(*result)) {                                                              \
        break;                                                                       \
      }                                                                              \
    }                                                                                \
  }

    while (read_offset < num_rows) {
      auto i = read_offset;
      WRITE_FIXEDWIDTH(Type::SHUFFLE_1BYTE, uint8_t);
      WRITE_FIXEDWIDTH(Type::SHUFFLE_2BYTE, uint16_t);
      WRITE_FIXEDWIDTH(Type::SHUFFLE_4BYTE, uint32_t);
      WRITE_FIXEDWIDTH(Type::SHUFFLE_8BYTE, uint64_t);
      WRITE_FIXEDWIDTH(Type::SHUFFLE_BIT, bool);
      if (!src_binary_arr.empty()) {
        for (i = read_offset; i < num_rows; ++i) {
          auto result = pid_writer_[new_id[i]]->WriteBinary(src_binary_arr, i);
          RETURN_NOT_OK(result.status());
          if (!(*result)) {
            break;
          }
        }
      }
      if (!src_large_binary_arr.empty()) {
        for (i = read_offset; i < num_rows; ++i) {
          auto result = pid_writer_[new_id[i]]->WriteLargeBinary(src_large_binary_arr, i);
          RETURN_NOT_OK(result.status());
          if (!(*result)) {
            break;
          }
        }
      }
      read_offset = i;
    }
#undef WRITE_FIXEDWIDTH

    return arrow::Status::OK();
  }

  arrow::Status Stop() {
    // write final record batch
    for (const auto& writer : pid_writer_) {
      RETURN_NOT_OK(writer->Stop());
    }
    std::sort(std::begin(temp_files), std::end(temp_files));
    return arrow::Status::OK();
  }

  arrow::Result<int64_t> TotalBytesWritten() {
    int64_t res = 0;
    for (const auto& writer : pid_writer_) {
      ARROW_ASSIGN_OR_RAISE(auto bytes, writer->BytesWritten());
      res += bytes;
    }
    return res;
  }

  Type::typeId column_type_id(int i) const { return column_type_id_[i]; }

  void set_buffer_size(int64_t buffer_size) { buffer_size_ = buffer_size; }

  void set_compression_codec(arrow::Compression::type compression_codec) { compression_codec_ = compression_codec; }

  std::shared_ptr<arrow::Schema> schema() const { return schema_; }

  std::shared_ptr<PartitionWriter> writer(int32_t pid) {
    if (pid_to_new_id_.find(pid) == pid_to_new_id_.end()) {
      return nullptr;
    }
    return pid_writer_[pid_to_new_id_[pid]];
  }

  std::vector<std::pair<int32_t, std::string>> temp_files;

 private:
  std::shared_ptr<arrow::Schema> schema_;

  // writer_schema_ removes the first field of schema_ which indicates the partition id
  std::shared_ptr<arrow::Schema> writer_schema_;

  int32_t num_partitiions_ = 0;
  Type::typeId last_type_;
  std::vector<Type::typeId> column_type_id_;
  std::unordered_map<int32_t, int32_t> pid_to_new_id_;
  std::vector<std::shared_ptr<PartitionWriter>> pid_writer_;

  int64_t buffer_size_ = kDefaultSplitterBufferSize;
  arrow::Compression::type compression_codec_ = arrow::Compression::UNCOMPRESSED;

  std::unique_ptr<arrow::fs::FileSystem> fs_;
  std::unique_ptr<arrow::internal::TemporaryDir> fs_base_dir_;
};

arrow::Result<std::shared_ptr<Splitter>> Splitter::Make(
    const std::shared_ptr<arrow::Schema>& schema) {
  std::shared_ptr<Splitter> ptr(new Splitter(schema));
  RETURN_NOT_OK(ptr->impl_->Init());
  return ptr;
}

Splitter::Splitter(const std::shared_ptr<arrow::Schema>& schema) {
  impl_.reset(new Impl(schema));
}

std::shared_ptr<arrow::Schema> Splitter::schema() const { return impl_->schema(); }

Type::typeId Splitter::column_type_id(int i) const { return impl_->column_type_id(i); }

arrow::Status Splitter::Split(const arrow::RecordBatch& rb) { return impl_->Split(rb); }

std::shared_ptr<PartitionWriter> Splitter::writer(int32_t pid) {
  return impl_->writer(pid);
}

arrow::Status Splitter::Stop() { return impl_->Stop(); }

const std::vector<std::pair<int32_t, std::string>>& Splitter::GetPartitionFileInfo()
    const {
  return impl_->temp_files;
}

void Splitter::set_buffer_size(int64_t buffer_size) {
  impl_->set_buffer_size(buffer_size);
}

void Splitter::set_compression_codec(arrow::Compression::type compression_codec) {
  impl_->set_compression_codec(compression_codec);
}

arrow::Result<int64_t> Splitter::TotalBytesWritten() {
  return impl_->TotalBytesWritten();
}

Splitter::~Splitter() = default;

}  // namespace shuffle
}  // namespace sparkcolumnarplugin
