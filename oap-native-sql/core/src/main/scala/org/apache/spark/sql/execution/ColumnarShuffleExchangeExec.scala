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

package org.apache.spark.sql.execution

import java.util.Random

import com.intel.sparkColumnarPlugin.vectorized.{
  ArrowColumnarBatchSerializer,
  ArrowWritableColumnVector
}
import org.apache.arrow.vector.types.pojo.Schema
import org.apache.arrow.vector.{FieldVector, IntVector}
import org.apache.spark.internal.Logging
import org.apache.spark.rdd.RDD
import org.apache.spark.serializer.Serializer
import org.apache.spark.shuffle.ColumnarShuffleDependency
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.codegen.LazilyGeneratedOrdering
import org.apache.spark.sql.catalyst.expressions.{
  Attribute,
  AttributeReference,
  BoundReference,
  UnsafeProjection
}
import org.apache.spark.sql.catalyst.plans.physical._
import org.apache.spark.sql.execution.exchange.ShuffleExchangeExec
import org.apache.spark.sql.execution.exchange.ShuffleExchangeExec.createShuffleWriteProcessor
import org.apache.spark.sql.execution.metric.{
  SQLMetric,
  SQLMetrics,
  SQLShuffleReadMetricsReporter,
  SQLShuffleWriteMetricsReporter
}
import org.apache.spark.sql.internal.SQLConf
import org.apache.spark.sql.types.{IntegerType, StructField, StructType}
import org.apache.spark.sql.util.ArrowUtils
import org.apache.spark.sql.vectorized.ColumnarBatch
import org.apache.spark.util.MutablePair
import org.apache.spark._

import scala.collection.JavaConverters._

class ColumnarShuffleExchangeExec(
    override val outputPartitioning: Partitioning,
    child: SparkPlan,
    canChangeNumPartitions: Boolean = true)
    extends ShuffleExchangeExec(outputPartitioning, child, canChangeNumPartitions) {

  private lazy val writeMetrics =
    SQLShuffleWriteMetricsReporter.createShuffleWriteMetrics(sparkContext)
  private lazy val readMetrics =
    SQLShuffleReadMetricsReporter.createShuffleReadMetrics(sparkContext)
  override lazy val metrics: Map[String, SQLMetric] = Map(
    "dataSize" -> SQLMetrics
      .createSizeMetric(sparkContext, "data size")) ++ readMetrics ++ writeMetrics

  override def nodeName: String = "ColumnarExchange"

  override def supportsColumnar: Boolean = true

  // Disable code generation
  @transient lazy val inputColumnarRDD: RDD[ColumnarBatch] = child.executeColumnar()

  private val serializer: Serializer = new ArrowColumnarBatchSerializer(longMetric("dataSize"))

  @transient
  lazy val columnarShuffleDependency: ShuffleDependency[Int, ColumnarBatch, ColumnarBatch] = {
    ColumnarShuffleExchangeExec.prepareShuffleDependency(
      inputColumnarRDD,
      child.output,
      outputPartitioning,
      serializer,
      writeMetrics)
  }

  def createColumnarShuffledRDD(
      partitionStartIndices: Option[Array[Int]]): ShuffledColumnarBatchRDD = {
    new ShuffledColumnarBatchRDD(columnarShuffleDependency, readMetrics, partitionStartIndices)
  }

  private var cachedShuffleRDD: ShuffledColumnarBatchRDD = null

  override def doExecuteColumnar(): RDD[ColumnarBatch] = {
    if (cachedShuffleRDD == null) {
      cachedShuffleRDD = createColumnarShuffledRDD(None)
    }
    cachedShuffleRDD
  }
}

object ColumnarShuffleExchangeExec {

  private def fromAttributes(attributes: Seq[Attribute]): StructType =
    StructType(attributes.map(a => StructField(a.name, a.dataType, a.nullable, a.metadata)))

  def prepareShuffleDependency(
      rdd: RDD[ColumnarBatch],
      outputAttributes: Seq[Attribute],
      newPartitioning: Partitioning,
      serializer: Serializer,
      writeMetrics: Map[String, SQLMetric])
    : ShuffleDependency[Int, ColumnarBatch, ColumnarBatch] = {

    val arrowSchema: Schema =
      ArrowUtils.toArrowSchema(
        fromAttributes(
          AttributeReference("pid", IntegerType, nullable = false)() +: outputAttributes),
        SQLConf.get.sessionLocalTimeZone)

    val part: Partitioner = newPartitioning match {
      case RoundRobinPartitioning(numPartitions) => new HashPartitioner(numPartitions)
      case HashPartitioning(_, n) =>
        new Partitioner {
          override def numPartitions: Int = n
          // For HashPartitioning, the partitioning key is already a valid partition ID, as we use
          // `HashPartitioning.partitionIdExpression` to produce partitioning key.
          override def getPartition(key: Any): Int = key.asInstanceOf[Int]
        }
      case RangePartitioning(sortingExpressions, numPartitions) =>
        // Extract only fields used for sorting to avoid collecting large fields that does not
        // affect sorting result when deciding partition bounds in RangePartitioner
        val rddForSampling = rdd.mapPartitionsInternal { iter =>
          // Internally, RangePartitioner runs a job on the RDD that samples keys to compute
          // partition bounds. To get accurate samples, we need to copy the mutable keys.
          iter.flatMap(batch => {
            val rows = batch.rowIterator.asScala
            val projection =
              UnsafeProjection.create(sortingExpressions.map(_.child), outputAttributes)
            val mutablePair = new MutablePair[InternalRow, Null]()
            rows.map(row => mutablePair.update(projection(row).copy(), null))
          })
        }
        // Construct ordering on extracted sort key.
        val orderingAttributes = sortingExpressions.zipWithIndex.map {
          case (ord, i) =>
            ord.copy(child = BoundReference(i, ord.dataType, ord.nullable))
        }
        implicit val ordering = new LazilyGeneratedOrdering(orderingAttributes)
        new RangePartitioner(
          numPartitions,
          rddForSampling,
          ascending = true,
          samplePointsPerPartitionHint = SQLConf.get.rangeExchangeSampleSizePerPartition)
      case SinglePartition =>
        new Partitioner {
          override def numPartitions: Int = 1
          override def getPartition(key: Any): Int = 0
        }
      case _ => sys.error(s"Exchange not implemented for $newPartitioning")
      // TODO: Handle BroadcastPartitioning.
    }

    val isRoundRobin = newPartitioning.isInstanceOf[RoundRobinPartitioning] &&
      newPartitioning.numPartitions > 1

    val rddWithDummyPartitionIds: RDD[Product2[Int, ColumnarBatch]] = {
      val isOrderSensitive = isRoundRobin && !SQLConf.get.sortBeforeRepartition

      rdd.mapPartitionsWithIndexInternal(
        (_, cbIterator) => {
          newPartitioning match {
            case SinglePartition =>
              // fixme hanle empty cb (cb.numRows == 0 || cb.numCols == 0)
              ClosablePairedColumnarBatchIterator(cbIterator.map(cb => {
                val pids = Array.fill(cb.numRows)(0)
                (0, pushFrontPartitionIds(pids, cb))
              }))
            case RoundRobinPartitioning(numPartitions) =>
              // Distributes elements evenly across output partitions, starting from a random partition.
              var position = new Random(TaskContext.get().partitionId()).nextInt(numPartitions)
              ClosablePairedColumnarBatchIterator(cbIterator.map(cb => {
                val pids = cb.rowIterator.asScala.map { _ =>
                  position += 1
                  part.getPartition(position)
                }.toArray
                (0, pushFrontPartitionIds(pids, cb))
              }))
            case h: HashPartitioning =>
              val projection =
                UnsafeProjection.create(h.partitionIdExpression :: Nil, outputAttributes)
              ClosablePairedColumnarBatchIterator(cbIterator.map(cb => {
                val pids = cb.rowIterator.asScala.map { row =>
                  part.getPartition(projection(row).getInt(0))
                }.toArray
                (0, pushFrontPartitionIds(pids, cb))
              }))
            case RangePartitioning(sortingExpressions, _) =>
              val projection =
                UnsafeProjection.create(sortingExpressions.map(_.child), outputAttributes)
              ClosablePairedColumnarBatchIterator(cbIterator.map(cb => {
                val pids = cb.rowIterator.asScala.map { row =>
                  part.getPartition(projection(row))
                }.toArray
                (0, pushFrontPartitionIds(pids, cb))
              }))
            case _ => sys.error(s"Exchange not implemented for $newPartitioning")
          }
        },
        isOrderSensitive = isOrderSensitive)
    }

    val dependency =
      new ColumnarShuffleDependency[Int, ColumnarBatch, ColumnarBatch](
        rddWithDummyPartitionIds,
        new PartitionIdPassthrough(newPartitioning.numPartitions),
        serializer,
        shuffleWriterProcessor = createShuffleWriteProcessor(writeMetrics),
        serializedSchema = arrowSchema.toByteArray)

    dependency
  }

  def pushFrontPartitionIds(partitionIds: Seq[Int], cb: ColumnarBatch): ColumnarBatch = {
    val length = partitionIds.length

    val vectors = (0 until cb.numCols()).map { idx =>
      val vector = cb
        .column(idx)
        .asInstanceOf[ArrowWritableColumnVector]
        .getValueVector
        .asInstanceOf[FieldVector]
      vector.setValueCount(length)
      vector
    }
    val pidVec = new IntVector("pid", vectors(0).getAllocator)

    pidVec.allocateNew(length)
    (0 until length).foreach { i =>
      pidVec.set(i, partitionIds(i))
    }
    pidVec.setValueCount(length)

    val newVectors = ArrowWritableColumnVector.loadColumns(length, (pidVec +: vectors).asJava)
    new ColumnarBatch(newVectors.toArray, cb.numRows)
  }
}

case class ClosablePairedColumnarBatchIterator(iter: Iterator[(Int, ColumnarBatch)])
    extends Iterator[(Int, ColumnarBatch)]
    with Logging {

  private var cur: (Int, ColumnarBatch) = _

  TaskContext.get().addTaskCompletionListener[Unit] { _ =>
    closeCurrentBatch()
  }

  private def closeCurrentBatch(): Unit = {
    if (cur != null) {
      logInfo("Shuffle ColumnarBatch closed")
      cur match { case (_, cb: ColumnarBatch) => cb.close() }
      cur = null
    }
  }

  override def hasNext: Boolean = {
    iter.hasNext
  }

  override def next(): (Int, ColumnarBatch) = {
    closeCurrentBatch()
    if (iter.hasNext) {
      cur = iter.next()
      cur
    } else Iterator.empty.next()
  }
}
