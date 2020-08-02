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

#include "shuffle/splitter.h"
#include <arrow/buffer_builder.h>
#include <arrow/filesystem/filesystem.h>
#include <arrow/filesystem/localfs.h>
#include <arrow/filesystem/path_util.h>
#include <arrow/type.h>
#include <arrow/util/io_util.h>
#include "shuffle/partition_writer.h"
#include "shuffle/utils.h"

#include <algorithm>
#include <memory>
#include <utility>
#include "shuffle/partitioning/single_splitter.h"

namespace sparkcolumnarplugin {
namespace shuffle {

// std::shared_ptr<PartitionWriter> Splitter::writer(int32_t pid) {
//  return impl_->writer(pid);
//}

}  // namespace shuffle
}  // namespace sparkcolumnarplugin
