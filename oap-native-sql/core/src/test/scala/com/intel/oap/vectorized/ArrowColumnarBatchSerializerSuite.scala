package com.intel.oap.vectorized

import java.io.{
  ByteArrayInputStream,
  ByteArrayOutputStream,
  FileInputStream,
  IOException,
  OutputStream
}

import org.apache.arrow.memory.{BufferAllocator, RootAllocator}
import org.apache.arrow.vector.VectorSchemaRoot
import org.apache.arrow.vector.ipc.{ArrowFileReader, ArrowStreamReader, ArrowStreamWriter}
import org.apache.spark.{SharedSparkContext, SparkFunSuite}
import org.apache.spark.sql.execution.metric.{SQLMetric, SQLMetrics}
import org.apache.spark.sql.test.SharedSparkSession
import org.apache.spark.sql.vectorized.ColumnarBatch

class ArrowColumnarBatchSerializerSuite extends SparkFunSuite with SharedSparkContext {

  private var metric: SQLMetric = _

  override def beforeEach() = {
    metric = SQLMetrics.createAverageMetric(sc, "test serializer")
  }

  test("deserialize null data") {
    val input = getTestResourcePath("test-data/native-splitter-output-all-null")
    val serializer = new ArrowColumnarBatchSerializer(metric).newInstance()
    val deserializedStream =
      serializer.deserializeStream(new FileInputStream(input))

    val kv = deserializedStream.asKeyValueIterator
    var length = 0
    kv.foreach {
      case (_, batch: ColumnarBatch) =>
        length += 1
        assert(batch.numRows == 4)
        assert(batch.numCols == 3)
        (0 until batch.numCols).foreach { i =>
          assert(
            batch
              .column(i)
              .asInstanceOf[ArrowWritableColumnVector]
              .getValueVector
              .getNullCount == batch.numRows)
        }
    }
    assert(length == 2)
    deserializedStream.close()
  }
}
