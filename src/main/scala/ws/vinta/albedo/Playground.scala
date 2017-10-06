package ws.vinta.albedo

import org.apache.spark.SparkConf
import org.apache.spark.ml.Transformer
import org.apache.spark.sql.SparkSession
import ws.vinta.albedo.recommenders._
import ws.vinta.albedo.utils.DatasetUtils._

object Playground {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf()
    conf.set("spark.driver.memory", "4g")
    conf.set("spark.executor.memory", "12g")
    conf.set("spark.executor.cores", "4")

    implicit val spark: SparkSession = SparkSession
      .builder()
      .appName("Playground")
      .config(conf)
      .getOrCreate()

    import spark.implicits._

    val sc = spark.sparkContext
    sc.setCheckpointDir("./spark-data/checkpoint")

    // Play

    val rawStarringDS = loadRawStarringDS()
    rawStarringDS.cache()

    val Array(_, testDF) = rawStarringDS.randomSplit(Array(0.8, 0.2))

    val testUserDF = testDF.select($"user_id").distinct()
    testUserDF.cache()

    val topK = 15

    val alsRecommender = new ALSRecommender()
      .setUserCol("user_id")
      .setItemCol("repo_id")
      .setTopK(topK)

    val recommenders = Array(alsRecommender)
    val userRecommendedItemDF = recommenders
      .map((recommender: Transformer) => recommender.transform(testUserDF))
      .reduce(_ union _)

    userRecommendedItemDF.show(false)

    spark.stop()
  }
}