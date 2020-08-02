package com.intel.oap.vectorized

case class PartitioningJniBridge(
    name: String,
    numPartitions: Int,
    serializedExprList: Array[Byte] = null)
