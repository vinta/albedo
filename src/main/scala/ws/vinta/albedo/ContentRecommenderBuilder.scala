package ws.vinta.albedo

import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import ws.vinta.albedo.evaluators.RankingEvaluator
import ws.vinta.albedo.evaluators.RankingEvaluator._
import ws.vinta.albedo.recommenders.ContentRecommender
import ws.vinta.albedo.utils.DatasetUtils._

object ContentRecommenderBuilder {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf()
    if (scala.util.Properties.envOrElse("RUN_WITH_INTELLIJ", "false") == "true") {
      conf.setMaster("local[*]")
      conf.set("spark.driver.memory", "12g")
      //conf.setMaster("local-cluster[1, 3, 12288]")
      //conf.setJars(List("target/albedo-1.0.0-SNAPSHOT-uber.jar"))
      //conf.setMaster("spark://localhost:7077")
      //conf.set("spark.driver.memory", "2g")
      //conf.set("spark.executor.cores", "3")
      //conf.set("spark.executor.memory", "12g")
      //conf.setJars(List("target/albedo-1.0.0-SNAPSHOT-uber.jar"))
    }

    implicit val spark: SparkSession = SparkSession
      .builder()
      .appName("ContentRecommenderBuilder")
      .config(conf)
      .getOrCreate()

    import spark.implicits._

    // Load Data

    val rawStarringDS = loadRawStarringDS()

    // Split Data

    val Array(_, testDF) = rawStarringDS.randomSplit(Array(0.9, 0.1))

    val largeUserIds = testDF.select($"user_id").distinct().map(row => row.getInt(0)).collect().toList
    val sampledUserIds = scala.util.Random.shuffle(largeUserIds).take(250) :+ 652070
    val testUserDF = spark.createDataFrame(sampledUserIds.map(Tuple1(_)))
      .toDF("user_id")
      .cache()

    // Make Recommendations

    val topK = 30

    val contentRecommender = new ContentRecommender()
      .setUserCol("user_id")
      .setItemCol("repo_id")
      .setTopK(topK)
      .setEnableEvaluationMode(true)

    val userRecommendedItemDF = contentRecommender
      .recommendForUsers(testUserDF)
      .cache()

    userRecommendedItemDF
      .where($"user_id" === 652070)
      .show(false)

    // Evaluate the Model

    val userActualItemsDF = loadUserActualItemsDF(topK).cache()

    val userPredictedItemsDF = userRecommendedItemDF
      .transform(intoUserPredictedItems($"user_id", $"repo_id", $"score".desc, topK))
      .cache()

    val rankingEvaluator = new RankingEvaluator(userActualItemsDF)
      .setMetricName("NDCG@k")
      .setK(topK)
      .setUserCol("user_id")
      .setItemsCol("items")
    val metric = rankingEvaluator.evaluate(userPredictedItemsDF)
    println(s"${rankingEvaluator.getFormattedMetricName} = $metric")
    // NDCG@30 = 0.002559563451967487

    spark.stop()
  }
}