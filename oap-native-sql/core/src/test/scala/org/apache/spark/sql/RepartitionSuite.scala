package org.apache.spark.sql

import org.apache.spark.SparkConf
import org.apache.spark.sql.execution.columnar.InMemoryTableScanExec
import org.apache.spark.sql.execution.exchange.ShuffleExchangeExec
import org.apache.spark.sql.execution.{
  ColumnarShuffleExchangeExec,
  ColumnarToRowExec,
  RowToColumnarExec
}
import org.apache.spark.sql.test.SharedSparkSession

class RepartitionSuite extends QueryTest with SharedSparkSession {
  import testImplicits._

  override def sparkConf: SparkConf =
    super.sparkConf
      .setAppName("test repartition")
      .set("spark.sql.parquet.columnarReaderBatchSize", "4096")
      .set("spark.sql.sources.useV1SourceList", "avro")
      .set("spark.sql.join.preferSortMergeJoin", "false")
      .set("spark.sql.extensions", "com.intel.sparkColumnarPlugin.ColumnarPlugin")
      .set("spark.sql.execution.arrow.maxRecordsPerBatch", "4096")
      .set("spark.shuffle.manager", "org.apache.spark.shuffle.sort.ColumnarShuffleManager")
//      .set("spark.shuffle.compress", "false")
      .set("spark.eventLog.enabled", "true")
      .set("spark.eventLog.dir", "file:///home/mr/spark/sparklog")

  def checkCoulumnarExec(data: DataFrame) = {
    val found = data.queryExecution.executedPlan
      .collect {
        case r2c: RowToColumnarExec => 1
        case c2r: ColumnarToRowExec => 10
        case exc: ColumnarShuffleExchangeExec => 100
      }
      .distinct
      .sum
    assert(found == 110)
  }

  def withInput(input: DataFrame)(
      transformation: Option[DataFrame => DataFrame],
      repartition: DataFrame => DataFrame): Unit = {
    val expected = transformation.getOrElse(identity[DataFrame](_))(input)
    val data = repartition(expected)
    checkCoulumnarExec(data)
    checkAnswer(data, expected)
  }

  lazy val input: DataFrame = Seq((1, "1"), (2, "20"), (3, "300")).toDF("id", "val")

  def withTransformationAndRepartition(
      transformation: DataFrame => DataFrame,
      repartition: DataFrame => DataFrame): Unit =
    withInput(input)(Some(transformation), repartition)

  def withRepartition: (DataFrame => DataFrame) => Unit = withInput(input)(None, _)
}

class SmallDataRepartitionSuite extends RepartitionSuite {
  import testImplicits._

  test("test round robin partitioning") {
    withRepartition(df => df.repartition(2))
  }

  test("test hash partitioning") {
    withRepartition(df => df.repartition('id))
  }

  test("test range partitioning") {
    withRepartition(df => df.repartitionByRange('id))
  }

  ignore("test cached repartiiton") {
    val data = input.cache.repartition(2)

    val found = data.queryExecution.executedPlan.collect {
      case cache: InMemoryTableScanExec => 1
      case c2r: ColumnarToRowExec => 10
      case exc: ColumnarShuffleExchangeExec => 100
    }.sum
    assert(found == 111)

    checkAnswer(data, input)
  }
}

class TPCHTableRepartitionSuite extends RepartitionSuite {
  import testImplicits._

  val filePath = getClass.getClassLoader
    .getResource("part-00000-d648dd34-c9d2-4fe9-87f2-770ef3551442-c000.snappy.parquet")
    .getFile

  override lazy val input = spark.read.parquet(filePath)

  test("test tpch round robin partitioning") {
    withRepartition(df => df.repartition(2))
  }

  test("test tpch hash partitioning") {
    withRepartition(df => df.repartition('n_nationkey))
  }

  test("test tpch range partitioning") {
    withRepartition(df => df.repartitionByRange('n_name))
  }

  test("test tpch sum after repartition") {
    withTransformationAndRepartition(
      df => df.groupBy("n_regionkey").agg(Map("n_nationkey" -> "sum")),
      df => df.repartition(2))
  }

  test("tpch q3") {
    val databaseName = scala.util.Properties.envOrElse("databaseName", "tpch1_d4d_nopart")
    spark.read
      .parquet(s"hdfs://127.0.0.1:9000/${databaseName}/customer")
      .createOrReplaceTempView("customer")
    spark.read
      .parquet(s"hdfs://127.0.0.1:9000/${databaseName}/orders")
      .createOrReplaceTempView("orders")
    spark.read
      .parquet(s"hdfs://127.0.0.1:9000/${databaseName}/lineitem")
      .createOrReplaceTempView("lineitem")
    val q3 = spark.sql(s"""select l_orderkey, sum(l_extendedprice * (1 - l_discount)) as revenue,
o_orderdate, o_shippriority from customer, orders, lineitem where c_mktsegment
= 'BUILDING' and c_custkey = o_custkey and l_orderkey = o_orderkey and
o_orderdate < "1995-03-05" and l_shipdate > "1995-03-05" group by
l_orderkey, o_orderdate, o_shippriority order by revenue desc, o_orderdate""")
    q3.explain
    q3.collect
  }
}

class DisableColumnarShuffle extends SmallDataRepartitionSuite {
  override def sparkConf: SparkConf = {
    super.sparkConf
      .set("spark.shuffle.manager", "sort")
      .set("spark.sql.codegen.wholeStage", "true")
  }

  override def checkCoulumnarExec(data: DataFrame) = {
    val found = data.queryExecution.executedPlan
      .collect {
        case c2r: ColumnarToRowExec => 1
        case exc: ColumnarShuffleExchangeExec => 10
        case exc: ShuffleExchangeExec => 100
      }
      .distinct
      .sum
    assert(found == 101)
  }
}
