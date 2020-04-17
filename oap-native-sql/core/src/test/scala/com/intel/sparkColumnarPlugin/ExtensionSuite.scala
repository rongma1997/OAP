package com.intel.sparkColumnarPlugin

import org.apache.spark.sql.execution.columnar.InMemoryTableScanExec
import org.apache.spark.sql.execution.{
  ColumnarShuffleExchangeExec,
  ColumnarToRowExec,
  RowToColumnarExec
}
import org.apache.spark.sql.{Row, SparkSession}
import org.scalatest.FunSuite

class ExtensionSuite extends FunSuite {

  private def stop(spark: SparkSession): Unit = {
    spark.stop()
    SparkSession.clearActiveSession()
    SparkSession.clearDefaultSession()
  }

  test("test round robin partitioning") {
    val session = SparkSession
      .builder()
      .master("local[1]")
      .config("org.apache.spark.example.columnar.enabled", value = true)
      .config("spark.sql.columnVector.arrow.enabled", value = true)
      .config("spark.sql.extensions", "com.intel.sparkColumnarPlugin.ColumnarPlugin")
      .config("spark.shuffle.manager", "org.apache.spark.shuffle.ColumnarShuffleManager")
      .appName("test round robin partitioning")
      .getOrCreate()

    try {
      import session.sqlContext.implicits._

      val input = Seq((1, 100), (2, 200), (3, 300))
      val data = input.toDF("id", "vals").repartition(2)

      val found = data.queryExecution.executedPlan.collect {
        case r2c: RowToColumnarExec => 1
        case c2r: ColumnarToRowExec => 10
        case exc: ColumnarShuffleExchangeExec => 100
      }.sum
      assert(found == 111)

      val result = data.collect()
      assert(result.toSet.equals(input.map { case (a, b) => Row(a, b) }.toSet))
    } finally {
      stop(session)
    }
  }

  test("test hash partitioning") {
    val session = SparkSession
      .builder()
      .master("local[1]")
      .config("org.apache.spark.example.columnar.enabled", value = true)
      .config("spark.sql.columnVector.arrow.enabled", value = true)
      .config("spark.sql.extensions", "com.intel.sparkColumnarPlugin.ColumnarPlugin")
      .config("spark.shuffle.manager", "org.apache.spark.shuffle.ColumnarShuffleManager")
      .appName("test hash partitioning")
      .getOrCreate()

    try {
      import session.sqlContext.implicits._

      val input = Seq((1, 100), (2, 200), (3, 300))
      val data = input.toDF("id", "vals").repartition('id)

      val found = data.queryExecution.executedPlan.collect {
        case r2c: RowToColumnarExec => 1
        case c2r: ColumnarToRowExec => 10
        case exc: ColumnarShuffleExchangeExec => 100
      }.sum
      assert(found == 111)

      val result = data.collect()
      assert(result.toSet.equals(input.map { case (a, b) => Row(a, b) }.toSet))
    } finally {
      stop(session)
    }
  }

  test("test range partitioning") {
    val session = SparkSession
      .builder()
      .master("local[1]")
      .config("org.apache.spark.example.columnar.enabled", value = true)
      .config("spark.sql.columnVector.arrow.enabled", value = true)
      .config("spark.sql.extensions", "com.intel.sparkColumnarPlugin.ColumnarPlugin")
      .config("spark.shuffle.manager", "org.apache.spark.shuffle.ColumnarShuffleManager")
      .appName("test range partitioning")
      .getOrCreate()

    try {
      import session.sqlContext.implicits._

      val input = Seq((1, 100), (2, 200), (3, 300))
      val data = input.toDF("id", "vals").repartitionByRange('id)

      val found = data.queryExecution.executedPlan.collect {
        case r2c: RowToColumnarExec => 1
        case c2r: ColumnarToRowExec => 10
        case exc: ColumnarShuffleExchangeExec => 100
      }.sum
      assert(found == 111)

      val result = data.collect()
      assert(result.toSet.equals(input.map { case (a, b) => Row(a, b) }.toSet))
    } finally {
      stop(session)
    }
  }

  ignore("test sum after repartition") {
    val session = SparkSession
      .builder()
      .master("local[1]")
      .config("org.apache.spark.example.columnar.enabled", value = true)
      .config("spark.sql.columnVector.arrow.enabled", value = true)
      .config("spark.sql.extensions", "com.intel.sparkColumnarPlugin.ColumnarPlugin")
      .config("spark.shuffle.manager", "org.apache.spark.shuffle.ColumnarShuffleManager")
      .appName("test sum after repartition")
      .getOrCreate()

    try {
      val filePath = "/tpch/lineitem"
      val sumFields = ("l_orderkey" :: "l_partkey" :: "l_suppkey" :: Nil).map(_ -> "sum").toMap

      val df = session.read.parquet(filePath)
      val data = df.agg(sumFields)
      val res = df.repartition(200).agg(sumFields)

      assert(data.first == res.first)
    } finally {
      session.stop()
    }
  }

  ignore("test tpch repartition variable") {
    val session = SparkSession
      .builder()
      .master("local[1]")
      .config("org.apache.spark.example.columnar.enabled", value = true)
      .config("spark.sql.columnVector.arrow.enabled", value = true)
      .config("spark.sql.extensions", "com.intel.sparkColumnarPlugin.ColumnarPlugin")
      .config("spark.shuffle.manager", "org.apache.spark.shuffle.ColumnarShuffleManager")
      .appName("test tpch repartition variable")
      .getOrCreate()

    try {
      val df = session.read.parquet("/tpch/lineitem")
      val data = df.select("l_linestatus").repartition(200)

      data.explain
      val found = data.queryExecution.executedPlan.collect {
        case r2c: RowToColumnarExec => 1
        case c2r: ColumnarToRowExec => 10
        case exc: ColumnarShuffleExchangeExec => 100
      }.sum
      assert(found == 110)

      data.foreach(_ => {})
    } finally {
      session.stop()
    }
  }

  test("test cached repartiiton") {
    val session = SparkSession
      .builder()
      .master("local[1]")
      .config("org.apache.spark.example.columnar.enabled", value = true)
      .config("spark.sql.columnVector.arrow.enabled", value = true)
      .config("spark.sql.extensions", "com.intel.sparkColumnarPlugin.ColumnarPlugin")
      .config("spark.shuffle.manager", "org.apache.spark.shuffle.ColumnarShuffleManager")
      .appName("test cached repartition")
      .getOrCreate()

    try {
      import session.sqlContext.implicits._

      val input = Seq((1, "1"), (2, "20"), (3, "300"))
      val data = input.toDF("id", "vals").cache.repartition(2)

      val found = data.queryExecution.executedPlan.collect {
        case cache: InMemoryTableScanExec => 1
        case c2r: ColumnarToRowExec => 10
        case exc: ColumnarShuffleExchangeExec => 100
      }.sum
      assert(found == 111)

      data.explain
      val result = data.collect
      assert(result.toSet.equals(input.map { case (a, b) => Row(a, b) }.toSet))
    } finally {
      stop(session)
    }
  }

  ignore("test cached tpch repartition") {
    val session = SparkSession
      .builder()
      .master("local[1]")
      .config("org.apache.spark.example.columnar.enabled", value = true)
      .config("spark.sql.columnVector.arrow.enabled", value = true)
      .config("spark.sql.extensions", "com.intel.sparkColumnarPlugin.ColumnarPlugin")
      .config("spark.shuffle.manager", "org.apache.spark.shuffle.ColumnarShuffleManager")
      .appName("test tpch repartition variable")
      .getOrCreate()

    try {
//      val filePath = "/tpch/lineitem"
      val filePath = s"${System.getProperty("user.home")}/people.csv"
      val selectField = "name"

      val df = session.read.csv(filePath)
      val data = df.select(selectField).repartition(2)

      data.explain
      val found = data.queryExecution.executedPlan.collect {
        case r2c: RowToColumnarExec => 1
        case c2r: ColumnarToRowExec => 10
        case exc: ColumnarShuffleExchangeExec => 100
      }.sum
      assert(found == 110)

      data.foreach(_ => {})
    } finally {
      session.stop()
    }
  }
}
