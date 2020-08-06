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

import com.google.common.collect.Lists
import com.intel.oap.expression.{
  CodeGeneration,
  ColumnarExpression,
  ColumnarExpressionConverter,
  ConverterUtils
}
import com.intel.oap.vectorized.{ArrowColumnarBatchSerializer, NativePartitioning}
import org.apache.arrow.gandiva.expression.TreeBuilder
import org.apache.arrow.vector.types.pojo.{Field, Schema}
import org.apache.spark._
import org.apache.spark.internal.Logging
import org.apache.spark.rdd.RDD
import org.apache.spark.serializer.Serializer
import org.apache.spark.shuffle.{ColumnarShuffleDependency, ShuffleHandle}
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.Attribute
import org.apache.spark.sql.catalyst.plans.physical._
import org.apache.spark.sql.execution.CoalesceExec.EmptyPartition
import org.apache.spark.sql.execution.exchange.ShuffleExchangeExec
import org.apache.spark.sql.execution.exchange.ShuffleExchangeExec.createShuffleWriteProcessor
import org.apache.spark.sql.execution.metric.{
  SQLMetric,
  SQLMetrics,
  SQLShuffleReadMetricsReporter,
  SQLShuffleWriteMetricsReporter
}
import org.apache.spark.sql.internal.SQLConf
import org.apache.spark.sql.vectorized.ColumnarBatch

import scala.collection.concurrent.TrieMap
import scala.collection.JavaConverters._
import scala.concurrent.Future

class ColumnarShuffleExchangeExec(
    override val outputPartitioning: Partitioning,
    child: SparkPlan,
    canChangeNumPartitions: Boolean = true)
    extends ShuffleExchangeExec(outputPartitioning, child, canChangeNumPartitions) {

  private lazy val writeMetrics =
    SQLShuffleWriteMetricsReporter.createShuffleWriteMetrics(sparkContext)
  override private[sql] lazy val readMetrics =
    SQLShuffleReadMetricsReporter.createShuffleReadMetrics(sparkContext)
  override lazy val metrics: Map[String, SQLMetric] = Map(
    "dataSize" -> SQLMetrics.createSizeMetric(sparkContext, "data size"),
    "splitTime" -> SQLMetrics.createNanoTimingMetric(sparkContext, "split time"),
    "computePidTime" -> SQLMetrics.createNanoTimingMetric(sparkContext, "compute pid time"),
    "totalTime" -> SQLMetrics.createNanoTimingMetric(sparkContext, "totaltime_shufflewrite"),
    "avgReadBatchNumRows" -> SQLMetrics
      .createAverageMetric(sparkContext, "avg read batch num rows")) ++ readMetrics ++ writeMetrics

  override def nodeName: String = "ColumnarExchange"

  override def supportsColumnar: Boolean = true

  private val serializer: Serializer = new ArrowColumnarBatchSerializer(
    longMetric("avgReadBatchNumRows"))

  @transient lazy val inputColumnarRDD: RDD[ColumnarBatch] = child.executeColumnar()

  // 'mapOutputStatisticsFuture' is only needed when enable AQE.
  @transient override lazy val mapOutputStatisticsFuture: Future[MapOutputStatistics] = {
    if (inputColumnarRDD.getNumPartitions == 0) {
      Future.successful(null)
    } else {
      sparkContext.submitMapStage(columnarShuffleDependency)
    }
  }

  /**
   * A [[ShuffleDependency]] that will partition rows of its child based on
   * the partitioning scheme defined in `newPartitioning`. Those partitions of
   * the returned ShuffleDependency will be the input of shuffle.
   */
  @transient
  lazy val columnarShuffleDependency: ShuffleDependency[Int, ColumnarBatch, ColumnarBatch] = {
    ColumnarShuffleExchangeExec.prepareShuffleDependency(
      inputColumnarRDD,
      child.output,
      outputPartitioning,
      serializer,
      writeMetrics,
      longMetric("dataSize"),
      longMetric("splitTime"),
      longMetric("computePidTime"),
      longMetric("totalTime"))
  }

  private var cachedShuffleRDD: ShuffledColumnarBatchRDD = _

  override def doExecuteColumnar(): RDD[ColumnarBatch] = {
    if (cachedShuffleRDD == null) {
      cachedShuffleRDD = new ShuffledColumnarBatchRDD(columnarShuffleDependency, readMetrics)
    }
    cachedShuffleRDD
  }

  // 'shuffleDependency' is only needed when enable AQE. Columnar shuffle will use 'columnarShuffleDependency'
  @transient
  override lazy val shuffleDependency: ShuffleDependency[Int, InternalRow, InternalRow] =
    new ShuffleDependency[Int, InternalRow, InternalRow](
      _rdd = new ColumnarShuffleExchangeExec.DummyPairRDDWithPartitions(
        sparkContext,
        inputColumnarRDD.getNumPartitions),
      partitioner = columnarShuffleDependency.partitioner) {

      override val shuffleId: Int = columnarShuffleDependency.shuffleId

      override val shuffleHandle: ShuffleHandle = columnarShuffleDependency.shuffleHandle
    }
}

object ColumnarShuffleExchangeExec extends Logging {

  val exchanges = new TrieMap[ShuffleExchangeExec, ColumnarShuffleExchangeExec]()

  class DummyPairRDDWithPartitions(@transient private val sc: SparkContext, numPartitions: Int)
      extends RDD[Product2[Int, InternalRow]](sc, Nil) {

    override def getPartitions: Array[Partition] =
      Array.tabulate(numPartitions)(i => EmptyPartition(i))

    override def compute(
        split: Partition,
        context: TaskContext): Iterator[Product2[Int, InternalRow]] = {
      throw new UnsupportedOperationException
    }
  }

  def prepareShuffleDependency(
      rdd: RDD[ColumnarBatch],
      outputAttributes: Seq[Attribute],
      newPartitioning: Partitioning,
      serializer: Serializer,
      writeMetrics: Map[String, SQLMetric],
      dataSize: SQLMetric,
      splitTime: SQLMetric,
      computePidTime: SQLMetric,
      totalTime: SQLMetric): ShuffleDependency[Int, ColumnarBatch, ColumnarBatch] = {

    val schemaFields = outputAttributes.map(attr => {
      Field
        .nullable(s"${attr.name}#${attr.exprId.id}", CodeGeneration.getResultType(attr.dataType))
    })
    val arrowSchema = new Schema(schemaFields.asJava)

    val nativePartitioning: NativePartitioning = newPartitioning match {
      case SinglePartition => new NativePartitioning("single", 1)
      case RoundRobinPartitioning(n) => new NativePartitioning("rr", n)
      case HashPartitioning(exprs, n) =>
        val gandivaExprs = exprs.zipWithIndex.map {
          case (expr, i) =>
            val columnarExpr = ColumnarExpressionConverter
              .replaceWithColumnarExpression(expr)
              .asInstanceOf[ColumnarExpression]
            val input: java.util.List[Field] = Lists.newArrayList()
            val (treeNode, resultType) = columnarExpr.doColumnarCodeGen(input)
            val attr = ConverterUtils.getAttrFromExpr(expr)
            val field = Field
              .nullable(
                s"${attr.name}#${attr.exprId.id}",
                CodeGeneration.getResultType(attr.dataType))
            TreeBuilder.makeExpression(treeNode, field)
        }
        new NativePartitioning("hash", n, ConverterUtils.getExprListBytesBuf(gandivaExprs.toList))
      // range partitioning fall back to row-based partition id computation
      case RangePartitioning(orders, n) =>
        val gandivaExprs = orders.zipWithIndex.map {
          case (order, i) =>
            val columnarExpr = ColumnarExpressionConverter
              .replaceWithColumnarExpression(order.child)
              .asInstanceOf[ColumnarExpression]
            val input: java.util.List[Field] = Lists.newArrayList()
            val (treeNode, resultType) = columnarExpr.doColumnarCodeGen(input)
            val attr = ConverterUtils.getAttrFromExpr(order.child)
            val field = Field
              .nullable(
                s"${attr.name}#${attr.exprId.id}",
                CodeGeneration.getResultType(attr.dataType))
            TreeBuilder.makeExpression(treeNode, field)
        }
        new NativePartitioning(
          "range",
          n,
          ConverterUtils.getExprListBytesBuf(gandivaExprs.toList))
    }

    val isRoundRobin = newPartitioning.isInstanceOf[RoundRobinPartitioning] &&
      newPartitioning.numPartitions > 1

    // RDD passed to ShuffleDependency should be the form of key-value pairs.
    // ColumnarShuffleWriter will compute ids from ColumnarBatch on native side other than read the "key" part.
    // Thus in Columnar Shuffle we never use the "key" part.
    val rddWithDummyKey: RDD[Product2[Int, ColumnarBatch]] = {
      val isOrderSensitive = isRoundRobin && !SQLConf.get.sortBeforeRepartition
      rdd.mapPartitionsWithIndexInternal(
        (_, cbIter) =>
          cbIter.map { cb =>
            (0, cb)
        },
        isOrderSensitive = isOrderSensitive)
    }

    val dependency =
      new ColumnarShuffleDependency[Int, ColumnarBatch, ColumnarBatch](
        rddWithDummyKey,
        new PartitionIdPassthrough(newPartitioning.numPartitions),
        serializer,
        shuffleWriterProcessor = createShuffleWriteProcessor(writeMetrics),
        serializedSchema = ConverterUtils.getSchemaBytesBuf(arrowSchema),
        nativePartitioning = nativePartitioning,
        dataSize = dataSize,
        computePidTime = computePidTime,
        splitTime = splitTime,
        totalTime = totalTime)

    dependency
  }
}
