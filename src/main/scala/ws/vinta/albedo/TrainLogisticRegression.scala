package ws.vinta.albedo

import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

import ws.vinta.albedo.schemas.starringSchema
import ws.vinta.albedo.utils.loadRawData

object TrainLogisticRegression {
  val appName = "TrainLogisticRegression"

  def main(args: Array[String]): Unit = {
    val activeUser = args(1)
    println(activeUser)

    val conf = new SparkConf()
    implicit val spark = SparkSession
      .builder()
      .config(conf)
      .appName(appName)
      .getOrCreate()
    val sc = spark.sparkContext

    import spark.implicits._

    val testDF = loadRawData()
    testDF.printSchema()

    val list = Seq(
      (1, 1, 1, "2017-05-16 20:01:00.0"),
      (1, 2, 1, "2017-05-17 21:01:00.0"),
      (2, 1, 0, "2017-05-18 22:01:00.0")
    )
    val tempDF = list
      .toDF("user", "item", "star", "starred_at")
      .withColumn("starred_at", unix_timestamp(col("starred_at")).cast("timestamp"))
    val df = spark.createDataFrame(tempDF.rdd, starringSchema)
    df.show()
    df.printSchema()

    spark.stop()
  }
}