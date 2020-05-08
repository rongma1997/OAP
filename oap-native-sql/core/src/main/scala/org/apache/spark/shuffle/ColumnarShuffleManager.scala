package org.apache.spark.shuffle

import java.io.InputStream
import java.util.concurrent.ConcurrentHashMap

import org.apache.spark._
import org.apache.spark.internal.Logging
import org.apache.spark.serializer.SerializerManager
import org.apache.spark.storage.BlockId
import org.apache.spark.util.collection.OpenHashSet

class ColumnarShuffleManager(conf: SparkConf) extends ShuffleManager with Logging {

  /**
   * A mapping from shuffle ids to the number of mappers producing output for those shuffles.
   */
  private[this] val taskIdMapsForShuffle = new ConcurrentHashMap[Int, OpenHashSet[Long]]()

  override val shuffleBlockResolver = new IndexShuffleBlockResolver(conf)

  /**
   * Obtains a [[ShuffleHandle]] to pass to tasks.
   */
  override def registerShuffle[K, V, C](
      shuffleId: Int,
      dependency: ShuffleDependency[K, V, C]): ShuffleHandle = {
    new ColumnarShuffleHandle[K, V](
      shuffleId,
      dependency.asInstanceOf[ShuffleDependency[K, V, V]])
  }

  /** Get a writer for a given partition. Called on executors by map tasks. */
  override def getWriter[K, V](
      handle: ShuffleHandle,
      mapId: Long,
      context: TaskContext,
      metrics: ShuffleWriteMetricsReporter): ShuffleWriter[K, V] = {
    val mapTaskIds =
      taskIdMapsForShuffle.computeIfAbsent(handle.shuffleId, _ => new OpenHashSet[Long](16))
    mapTaskIds.synchronized { mapTaskIds.add(context.taskAttemptId()) }
    handle match {
      case columnarShuffleHandle: ColumnarShuffleHandle[K @unchecked, V @unchecked] =>
        new ColumnarShuffleWriter(shuffleBlockResolver, columnarShuffleHandle, mapId, metrics)
      case other =>
        throw new UnsupportedOperationException(s"Unsupported ShuffleHandle ${other.getClass}")
    }
  }

  /**
   * Get a reader for a range of reduce partitions (startPartition to endPartition-1, inclusive).
   * Called on executors by reduce tasks.
   */
  override def getReader[K, C](
      handle: ShuffleHandle,
      startPartition: Int,
      endPartition: Int,
      context: TaskContext,
      metrics: ShuffleReadMetricsReporter): ShuffleReader[K, C] = {
    val serializerManager = new SerializerManager(
      SparkEnv.get.serializer,
      SparkEnv.get.conf,
      SparkEnv.get.securityManager.getIOEncryptionKey()) {
      // Bypass the shuffle read compression
      override def wrapStream(blockId: BlockId, s: InputStream): InputStream = {
        wrapForEncryption(s)
      }
    }
    val blocksByAddress = SparkEnv.get.mapOutputTracker
      .getMapSizesByExecutorId(handle.shuffleId, startPartition, endPartition)
    new BlockStoreShuffleReader(
      handle.asInstanceOf[BaseShuffleHandle[K, _, C]],
      blocksByAddress,
      context,
      metrics,
      serializerManager)
  }

  /** Remove a shuffle's metadata from the ShuffleManager. */
  override def unregisterShuffle(shuffleId: Int): Boolean = {
    Option(taskIdMapsForShuffle.remove(shuffleId)).foreach { mapTaskIds =>
      mapTaskIds.iterator.foreach { mapId =>
        shuffleBlockResolver.removeDataByMap(shuffleId, mapId)
      }
    }
    true
  }

  /** Shut down this ShuffleManager. */
  override def stop(): Unit = {
    shuffleBlockResolver.stop()
  }

  override def getReaderForRange[K, C](
      handle: ShuffleHandle,
      startMapIndex: Int,
      endMapIndex: Int,
      startPartition: Int,
      endPartition: Int,
      context: TaskContext,
      metrics: ShuffleReadMetricsReporter): ShuffleReader[K, C] = {
    val blocksByAddress = SparkEnv.get.mapOutputTracker.getMapSizesByRange(
      handle.shuffleId,
      startMapIndex,
      endMapIndex,
      startPartition,
      endPartition)
    new BlockStoreShuffleReader(
      handle.asInstanceOf[BaseShuffleHandle[K, _, C]],
      blocksByAddress,
      context,
      metrics)
  }
}

private[spark] class ColumnarShuffleHandle[K, V](
    shuffleId: Int,
    dependency: ShuffleDependency[K, V, V])
    extends BaseShuffleHandle(shuffleId, dependency) {}
