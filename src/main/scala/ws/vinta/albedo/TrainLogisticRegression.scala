package ws.vinta.albedo

import org.apache.spark.SparkConf
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}
import ws.vinta.albedo.preprocessors.{NegativeGenerator, popularItemsBuilder, starringBuilder}
import ws.vinta.albedo.schemas.{PopularItem, Starring}
import ws.vinta.albedo.utils.CommonUtils

import scala.collection.mutable

object TrainLogisticRegression {
  val appName = "TrainLogisticRegression"

  def main(args: Array[String]): Unit = {
    val activeUser = args(1)
    println(activeUser)

    val conf = new SparkConf()
    conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")

    implicit val spark = SparkSession
      .builder()
      .config(conf)
      .appName(appName)
      .getOrCreate()

    implicit val sc = spark.sparkContext
    sc.setLogLevel("WARN")

    import spark.implicits._

    val rawDF: DataFrame = CommonUtils.loadRawData()
    rawDF.cache()

    val starringDS: Dataset[Starring] = starringBuilder.transform(rawDF).as[Starring]
    starringDS.show()
    println(starringDS.count())

    val popularItems: mutable.LinkedHashSet[Int] = popularItemsBuilder.transform(rawDF).as[PopularItem]
      .select("item")
      .map(r => r(0).asInstanceOf[Int])
      .collect()
      .to[mutable.LinkedHashSet]

    //val list = Seq(
    //  (1, 1, 1, "2017-05-16 20:01:00.0"),
    //  (1, 2, 1, "2017-05-17 21:01:00.0"),
    //  (1, 4, 1, "2017-05-17 21:01:00.0"),
    //  (2, 1, 1, "2017-05-18 22:01:00.0"),
    //  (3, 1, 1, "2017-05-10 22:01:00.0"),
    //  (3, 2, 1, "2017-05-10 22:01:00.0"),
    //  (3, 5, 1, "2017-05-10 22:01:00.0"),
    //  (3, 10, 1, "2017-05-10 22:01:00.0")
    //)
    //import org.apache.spark.sql.functions._
    //val starringDS = list
    //  .toDF("user", "item", "star", "starred_at")
    //  .withColumn("starred_at", unix_timestamp(col("starred_at")).cast("timestamp"))
    //  .as[Starring]
    //val popularItems = mutable.LinkedHashSet(1, 2, 7, 3, 10, 14, 4, 21, 9)

    val bcPopularItems = sc.broadcast(popularItems)
    val negativeGenerator = new NegativeGenerator(bcPopularItems)
    negativeGenerator
      .setNegativeValue(0)
      .setNegativePositiveRatio(1.0)
    val balancedDF: DataFrame = negativeGenerator.transform(starringDS)
    balancedDF.show()
    println(balancedDF.count())

    spark.stop()
  }
}