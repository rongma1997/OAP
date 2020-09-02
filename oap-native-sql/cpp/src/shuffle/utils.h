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

#include <iomanip>
#include <sstream>

#include <arrow/filesystem/filesystem.h>
#include <arrow/filesystem/localfs.h>
#include <arrow/filesystem/path_util.h>
#include <arrow/util/io_util.h>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>

namespace sparkcolumnarplugin {
namespace shuffle {

static std::string GenerateUUID() {
  boost::uuids::random_generator generator;
  return boost::uuids::to_string(generator());
}

// For test purpose
static arrow::Result<std::string> CreateTempShuffleDir() {
  bool exist = true;
  std::string dir;
  while (exist) {
    auto result = arrow::internal::TemporaryDir::Make("columnar-shuffle-");
    if (result.ok()) {
      dir = std::move(result.MoveValueUnsafe()->path().ToString());
      exist = false;
    } else if (!result.status().IsIOError()) {
      return result.status();
    }
  }
  return dir;
}

static arrow::Result<std::string> CreateTempShuffleFile(const std::string& dir) {
  if (dir.length() == 0) {
    return arrow::Status::Invalid("Failed to create spilled file, got empty path.");
  }

  auto fs = std::make_shared<arrow::fs::LocalFileSystem>();
  ARROW_ASSIGN_OR_RAISE(auto path_info, fs->GetFileInfo(dir));
  if (path_info.type() == arrow::fs::FileType::NotFound) {
    RETURN_NOT_OK(fs->CreateDir(dir, true));
  }

  bool exist = true;
  std::string file_path;
  while (exist) {
    file_path =
        arrow::fs::internal::ConcatAbstractPath(dir, "temp_shuffle_" + GenerateUUID());
    ARROW_ASSIGN_OR_RAISE(auto file_info, fs->GetFileInfo(file_path));
    if (file_info.type() == arrow::fs::FileType::NotFound) {
      exist = false;
      ARROW_ASSIGN_OR_RAISE(auto s, fs->OpenOutputStream(file_path));
      RETURN_NOT_OK(s->Close());
    }
  }
  return file_path;
}

static arrow::ipc::IpcWriteOptions SplitterIpcWriteOptions(
    arrow::Compression::type compression) {
  auto options = arrow::ipc::IpcWriteOptions::Defaults();
  options.compression = compression;
  options.use_threads = false;
  return options;
}

static arrow::Result<std::vector<Type::typeId>> ToSplitterTypeId(
    const std::vector<std::shared_ptr<arrow::Field>>& fields) {
  std::vector<Type::typeId> splitter_type_id;
  splitter_type_id.reserve(fields.size());
  std::pair<std::string, arrow::Type::type> field_type_not_implemented;

  std::transform(std::cbegin(fields), std::cend(fields),
                 std::back_inserter(splitter_type_id),
                 [&field_type_not_implemented](
                     const std::shared_ptr<arrow::Field>& field) -> Type::typeId {
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
                     case arrow::Date32Type::type_id:
                     case arrow::Time32Type::type_id:
                       return Type::SHUFFLE_4BYTE;
                     case arrow::Int64Type::type_id:
                     case arrow::UInt64Type::type_id:
                     case arrow::DoubleType::type_id:
                     case arrow::Date64Type::type_id:
                     case arrow::Time64Type::type_id:
                     case arrow::TimestampType::type_id:
                       return Type::SHUFFLE_8BYTE;
                     case arrow::BinaryType::type_id:
                     case arrow::StringType::type_id:
                       return Type::SHUFFLE_BINARY;
                     case arrow::LargeBinaryType::type_id:
                     case arrow::LargeStringType::type_id:
                       return Type::SHUFFLE_LARGE_BINARY;
                     case arrow::Decimal128Type::type_id:
                       return Type::SHUFFLE_DECIMAL128;
                     case arrow::NullType::type_id:
                       return Type::SHUFFLE_NULL;
                     default:
                       field_type_not_implemented =
                           std::make_pair(std::move(field->ToString()), arrow_type_id);
                       return Type::SHUFFLE_NOT_IMPLEMENTED;
                   }
                 });

  auto it = std::find(std::begin(splitter_type_id), std::end(splitter_type_id),
                      Type::SHUFFLE_NOT_IMPLEMENTED);
  if (it != std::end(splitter_type_id)) {
    RETURN_NOT_OK(arrow::Status::NotImplemented(
        "Field type not implemented: " + field_type_not_implemented.first +
        "\n arrow type id: " + std::to_string(field_type_not_implemented.second)));
  }
  return splitter_type_id;
}

}  // namespace shuffle
}  // namespace sparkcolumnarplugin
