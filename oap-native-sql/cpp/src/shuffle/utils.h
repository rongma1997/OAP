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

static arrow::Result<std::string> CreateRandomSubDir(const std::string& base_dir) {
  bool created = false;
  std::string random_dir;
  while(!created) {
    random_dir = arrow::fs::internal::ConcatAbstractPath(
        base_dir, GenerateUUID());
    ARROW_ASSIGN_OR_RAISE(
        created, arrow::internal::CreateDirTree(
        *arrow::internal::PlatformFilename::FromString(random_dir)));
  }
  return random_dir;
}

static arrow::Result<std::vector<std::string>> GetConfiguredLocalDirs() {
  auto joined_dirs_c = std::getenv("NATIVESQL_SPARK_LOCAL_DIRS");
  if (joined_dirs_c != nullptr && strcmp(joined_dirs_c, "") > 0) {
    auto joined_dirs = std::string(joined_dirs_c);
    std::string delimiter = ",";
    std::vector<std::string> dirs;

    size_t pos;
    std::string root_dir;
    while ((pos = joined_dirs.find(delimiter)) != std::string::npos) {
      root_dir = joined_dirs.substr(0, pos);
      if (root_dir.length() > 0) {
        dirs.push_back(root_dir);
      }
      joined_dirs.erase(0, pos + delimiter.length());
    }
    if (joined_dirs.length() > 0) {
      dirs.push_back(joined_dirs);
    }
    return dirs;
  } else {
    ARROW_ASSIGN_OR_RAISE(auto arrow_tmp_dir,
                          arrow::internal::TemporaryDir::Make("columnar-shuffle-"));
    return std::vector<std::string>{arrow_tmp_dir->path().ToString()};
  }
}

static const arrow::ipc::IpcWriteOptions GetIpcWriteOptions(arrow::Compression::type compression) {
  auto options = arrow::ipc::IpcWriteOptions::Defaults();
  options.compression = compression;
  options.use_threads = false;
}

static const arrow::Result<arrow::Compression::type> GetCompressionCodec() {
  auto codec_l = std::getenv("NATIVESQL_COMPRESSION_CODEC");
  auto compression_codec = arrow::Compression::UNCOMPRESSED;
  if (codec_l != nullptr) {
    std::string codec_u;
    std::transform(codec_l, codec_l + std::strlen(codec_l), std::back_inserter(codec_u),
                   ::toupper);

    ARROW_ASSIGN_OR_RAISE(compression_codec, arrow::util::Codec::GetCompressionType(codec_u))

    if (compression_codec == arrow::Compression::LZ4) {
      compression_codec = arrow::Compression::LZ4_FRAME;
    }
  }
  return compression_codec;
}

}  // namespace shuffle
}  // namespace sparkcolumnarplugin
