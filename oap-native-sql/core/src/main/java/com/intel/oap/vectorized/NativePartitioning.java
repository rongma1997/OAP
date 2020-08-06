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

package com.intel.oap.vectorized;

import java.io.Serializable;

/**
 * Hold partitioning parameters needed by native splitter
 */
public class NativePartitioning implements Serializable {
  String shortName;
  int numPartitions;
  byte[] serializedSchema;
  byte[] serializedExprList;

  public NativePartitioning(String shortName, int numPartitions, byte[] serialzedSchema, byte[] serializedExprList) {
    this.shortName = shortName;
    this.numPartitions = numPartitions;
    this.serializedExprList = serializedExprList;
  }

  public NativePartitioning(String shortName, int numPartitions, byte[] serializedSchema) {
    this(shortName, numPartitions, serializedSchema, null);
  }

  public String getShortName() {
    return shortName;
  }

  public int getNumPartitions() {
    return numPartitions;
  }

  public byte[] getSerializedExprList() {
    return serializedExprList;
  }
}
