package com.intel.sparkColumnarPlugin.execution

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.Attribute
import org.apache.spark.sql.execution.{SparkPlan, UnaryExecNode}
import org.apache.spark.sql.vectorized.ColumnarBatch

case class CoalesceBatchesExec(child: SparkPlan) extends UnaryExecNode {

  override def supportsColumnar: Boolean = true

  override def nodeName: String = "CoalesceBatches"

  override protected def doExecute(): RDD[InternalRow] = throw new UnsupportedOperationException

  override def doExecuteColumnar(): RDD[ColumnarBatch] = {
    child.executeColumnar()
  }

  override def output: Seq[Attribute] = child.output

}
