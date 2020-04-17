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

package com.intel.sparkColumnarPlugin.vectorized

import java.io._
import java.nio.ByteBuffer

import org.apache.arrow.memory.BufferAllocator
import org.apache.arrow.util.SchemaUtils
import org.apache.arrow.vector.ipc.ArrowStreamReader
import org.apache.arrow.vector.{VectorLoader, VectorSchemaRoot}
import org.apache.spark.SparkEnv
import org.apache.spark.internal.Logging
import org.apache.spark.serializer.{
  DeserializationStream,
  SerializationStream,
  Serializer,
  SerializerInstance
}
import org.apache.spark.sql.execution.metric.SQLMetric
import org.apache.spark.sql.internal.SQLConf
import org.apache.spark.sql.util.ArrowUtils
import org.apache.spark.sql.vectorized.{ColumnVector, ColumnarBatch}

import scala.collection.JavaConverters._
import scala.collection.mutable
import scala.collection.mutable.ListBuffer
import scala.reflect.ClassTag

class ArrowColumnarBatchSerializer(dataSize: SQLMetric) extends Serializer with Serializable {

  /** Creates a new [[SerializerInstance]]. */
  override def newInstance(): SerializerInstance =
    new ArrowColumnarBatchSerializerInstance(dataSize)
}

private class ArrowColumnarBatchSerializerInstance(dataSize: SQLMetric)
    extends SerializerInstance
    with Logging {

  override def deserializeStream(in: InputStream): DeserializationStream = {
    new DeserializationStream {

      private val columnBatchSize = SQLConf.get.columnBatchSize
      private val compressionEnabled =
        SparkEnv.get.conf.getBoolean("spark.shuffle.compress", true)
      private val compressionCodec = SparkEnv.get.conf.get("spark.io.compression.codec", "lz4")
      private val allocator: BufferAllocator = ArrowUtils.rootAllocator
        .newChildAllocator("ArrowColumnarBatch deserialize", 0, Long.MaxValue)

      private var reader: ArrowStreamReader = _
      private var root: VectorSchemaRoot = _
      private var vectors: Array[ColumnVector] = _
      private var batchLoaded = true

      private var jniWrapper: ShuffleDecompressionJniWrapper = _
      private var schemaHolderId: Long = 0
      private var vectorLoader: VectorLoader = _

      override def asIterator: Iterator[Any] = {
        // This method is never called by shuffle code.
        throw new UnsupportedOperationException
      }

      override def readKey[T: ClassTag](): T = {
        // We skipped serialization of the key in writeKey(), so just return a dummy value since
        // this is going to be discarded anyways.
        null.asInstanceOf[T]
      }

      @throws(classOf[EOFException])
      override def readValue[T: ClassTag](): T = {
        if (reader != null && batchLoaded) {
          batchLoaded = reader.loadNextBatch()
          if (batchLoaded) {
            assert(
              root.getRowCount <= columnBatchSize,
              "the number of loaded rows exceed the maximum columnar batch size")

            // jni call to decompress buffers
            if (compressionEnabled) {
              decompressVectors()
            }

            val batch = new ColumnarBatch(vectors, root.getRowCount)
            batch.asInstanceOf[T]
          } else {
            this.close()
            throw new EOFException
          }
        } else {
          reader = new ArrowStreamReader(in, allocator)
          try {
            root = reader.getVectorSchemaRoot
          } catch {
            case _: IOException =>
              this.close()
              throw new EOFException
          }
          vectors = ArrowWritableColumnVector
            .loadColumns(root.getRowCount, root.getFieldVectors)
            .toArray[ColumnVector]
          readValue()
        }
      }

      override def readObject[T: ClassTag](): T = {
        // This method is never called by shuffle code.
        throw new UnsupportedOperationException
      }

      override def close(): Unit = {
        if (reader != null) reader.close(false)
        if (allocator != null) allocator.close()
        if (jniWrapper != null) jniWrapper.close(schemaHolderId)
        in.close()
      }

      private def decompressVectors(): Unit = {
        if (jniWrapper == null) {
          jniWrapper = new ShuffleDecompressionJniWrapper
          schemaHolderId = jniWrapper.make(SchemaUtils.get.serialize(root.getSchema))
        }
        if (vectorLoader == null) {
          vectorLoader = new VectorLoader(root)
        }
        val bufAddrs = new ListBuffer[Long]()
        val bufSizes = new ListBuffer[Long]()
        val bufBS = mutable.BitSet()
        var bufIdx = 0

        root.getFieldVectors.asScala.foreach { vector =>
          if (vector.getNullCount == 0 || vector.getNullCount == vector.getValueCount) {
            bufBS.add(bufIdx)
          }
          vector.getBuffers(false).foreach { buffer =>
            bufAddrs += buffer.memoryAddress()
            // buffer.readableBytes() will return wrong readable length here since it is initialized by
            // data stored in IPC message header, which is not the actual compressed length
            bufSizes += buffer.capacity()
            bufIdx += 1
          }
        }

        val builder = jniWrapper.decompress(
          schemaHolderId,
          compressionCodec,
          root.getRowCount,
          bufAddrs.toArray,
          bufSizes.toArray,
          bufBS.toBitMask)
        val builerImpl = new ArrowRecordBatchBuilderImpl(builder)
        val decompressedRecordBatch = builerImpl.build
        // vectors previously loaded into the root are released by arrow when loading new vectors
        // so don't bother releasing/closing them here
        vectorLoader.load(decompressedRecordBatch)
      }
    }
  }

  // Columnar shuffle write process don't need this.
  override def serializeStream(s: OutputStream): SerializationStream =
    throw new UnsupportedOperationException

  // These methods are never called by shuffle code.
  override def serialize[T: ClassTag](t: T): ByteBuffer = throw new UnsupportedOperationException

  override def deserialize[T: ClassTag](bytes: ByteBuffer): T =
    throw new UnsupportedOperationException

  override def deserialize[T: ClassTag](bytes: ByteBuffer, loader: ClassLoader): T =
    throw new UnsupportedOperationException
}
