package ws.vinta.albedo

import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import ws.vinta.albedo.preprocessors.starringBuilder
import ws.vinta.albedo.preprocessors.popularItemsBuilder
import ws.vinta.albedo.utils.loadRawData
import ws.vinta.albedo.schemas.{PopularItem, Starring}

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

    import spark.implicits._

    val rawDF = loadRawData()
    rawDF.cache()

    val starringDS = starringBuilder.transform(rawDF).as[Starring]
    starringDS.show()

    val popularReposDS = popularItemsBuilder.transform(rawDF).as[PopularItem]
    popularReposDS.show()

    spark.stop()
  }
}