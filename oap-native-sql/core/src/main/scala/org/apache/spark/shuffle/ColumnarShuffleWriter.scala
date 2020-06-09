package org.apache.spark.shuffle

import java.io.{File, FileInputStream, FileOutputStream, IOException}
import java.nio.ByteBuffer

import com.google.common.annotations.VisibleForTesting
import com.google.common.io.Closeables
import com.intel.sparkColumnarPlugin.vectorized.ShuffleSplitterJniWrapper
import org.apache.arrow.util.SchemaUtils
import org.apache.arrow.vector.types.pojo.Schema
import org.apache.spark._
import org.apache.spark.internal.Logging
import org.apache.spark.scheduler.MapStatus
import org.apache.spark.sql.vectorized.ColumnarBatch
import org.apache.spark.util.Utils

import scala.collection.mutable.ListBuffer

class ColumnarShuffleWriter[K, V](
    shuffleBlockResolver: IndexShuffleBlockResolver,
    handle: BaseShuffleHandle[K, V, V],
    mapId: Long,
    writeMetrics: ShuffleWriteMetricsReporter)
    extends ShuffleWriter[K, V]
    with Logging {

  private val dep = handle.dependency

  private val conf = SparkEnv.get.conf

  private val blockManager = SparkEnv.get.blockManager

  // Are we in the process of stopping? Because map tasks can call stop() with success = true
  // and then call stop() with success = false if they get an exception, we want to make sure
  // we don't try deleting files, etc twice.
  private var stopping = false

  private var mapStatus: MapStatus = _

  private val transeferToEnabled = conf.getBoolean("spark.file.transferTo", true)
  private val compressionEnabled = conf.getBoolean("spark.shuffle.compress", true)
  private val compressionCodec = conf.get("spark.io.compression.codec", "lz4")

  private val jniWrapper = new ShuffleSplitterJniWrapper()

  private var nativeSplitter: Long = 0

  private var partitionLengths: List[Long] = _

  @throws[IOException]
  override def write(records: Iterator[Product2[K, V]]): Unit = {
    if (!records.hasNext) {
      partitionLengths = List[Long]()
      shuffleBlockResolver.writeIndexFileAndCommit(
        dep.shuffleId,
        mapId,
        partitionLengths.toArray,
        null)
      mapStatus = MapStatus(blockManager.shuffleServerId, partitionLengths.toArray, mapId)
      return
    }

    if (nativeSplitter == 0) {
      val schema: Schema = Schema.deserialize(
        ByteBuffer.wrap(dep.asInstanceOf[ColumnarShuffleDependency[K, V, V]].serializedSchema))
      nativeSplitter = jniWrapper.make(SchemaUtils.get.serialize(schema))
      if (compressionEnabled) {
        jniWrapper.setCompressionCodec(nativeSplitter, compressionCodec)
      }
    }

    while (records.hasNext) {
      val columnarBatch = records.next()._2.asInstanceOf[ColumnarBatch]

      val bufAddrs = new ListBuffer[Long]()
      val bufSizes = new ListBuffer[Long]()
      (0 until columnarBatch.numCols).foreach { idx =>
        val column = columnarBatch.column(idx).asInstanceOf[ArrowWritableColumnVector]
        column.getValueVector
          .getBuffers(false)
          .foreach { buffer =>
            bufAddrs += buffer.memoryAddress()
            bufSizes += buffer.readableBytes()
          }
      }

      val startTime = System.nanoTime()
      jniWrapper.split(nativeSplitter, columnarBatch.numRows, bufAddrs.toArray, bufSizes.toArray)
      writeMetrics.incWriteTime(System.nanoTime() - startTime)
      writeMetrics.incRecordsWritten(1)
    }

    val startTime = System.nanoTime()
    jniWrapper.stop(nativeSplitter)
    writeMetrics.incWriteTime(System.nanoTime() - startTime)
    writeMetrics.incBytesWritten(jniWrapper.getTotalBytesWritten(nativeSplitter))

    val output = shuffleBlockResolver.getDataFile(dep.shuffleId, mapId)
    val tmp = Utils.tempFileWith(output)
    try {
      partitionLengths = writePartitionedFile(tmp)
      shuffleBlockResolver.writeIndexFileAndCommit(
        dep.shuffleId,
        mapId,
        partitionLengths.toArray,
        tmp);
    } finally {
      if (tmp.exists() && !tmp.delete()) {
        logError(s"Error while deleting temp file ${tmp.getAbsolutePath()}")
      }
    }
    mapStatus = MapStatus(blockManager.shuffleServerId, partitionLengths.toArray, mapId)
  }

  override def stop(success: Boolean): Option[MapStatus] = {

    try {
      if (stopping) {
        None
      }
      stopping = true
      if (success) {
        Option(mapStatus)
      } else {
        None
      }
    } finally {
      // delete the temporary files hold by native splitter
      if (nativeSplitter != 0) {
        try {
          jniWrapper.getPartitionFileInfo(nativeSplitter).foreach { fileInfo =>
            {
              val pid = fileInfo.getPid()
              val file = new File(fileInfo.getFilePath())
              if (file.exists()) {
                if (!file.delete()) {
                  logError(s"Unable to delete file for partition ${pid}")
                }
              }
            }
          }
        } finally {
          jniWrapper.close(nativeSplitter)
          nativeSplitter = 0
        }
      }
    }
  }

  @throws[IOException]
  private def writePartitionedFile(outputFile: File): List[Long] = {

    val lengths = new Array[Long](dep.partitioner.numPartitions)
    val out = new FileOutputStream(outputFile, true)
    val writerStartTime = System.nanoTime()
    var threwException = true

    try {
      jniWrapper.getPartitionFileInfo(nativeSplitter).foreach { fileInfo =>
        {
          val pid = fileInfo.getPid
          val filePath = fileInfo.getFilePath

          val file = new File(filePath)
          if (file.exists()) {
            val in = new FileInputStream(file)
            var copyThrewException = true

            try {
              lengths(pid) = Utils.copyStream(in, out, false, transeferToEnabled)
              copyThrewException = false
            } finally {
              Closeables.close(in, copyThrewException)
            }
            if (!file.delete()) {
              logError(s"Unable to delete file for partition ${pid}")
            } else {
              logInfo(s"Deleting temporary shuffle file ${filePath} for partition ${pid}")
            }
          } else {
            logWarning(
              s"Native shuffle writer temporary file ${filePath} for partition ${pid} not exists")
          }
        }
      }
      threwException = false
    } finally {
      Closeables.close(out, threwException)
      writeMetrics.incWriteTime(System.nanoTime - writerStartTime)
    }
    lengths.toList
  }

  @VisibleForTesting
  def getPartitionLengths: List[Long] = partitionLengths

}
