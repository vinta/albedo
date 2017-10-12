package ws.vinta.albedo

import org.apache.spark.sql.SparkSession
import ws.vinta.albedo.evaluators.RankingEvaluator
import ws.vinta.albedo.evaluators.RankingEvaluator._
import ws.vinta.albedo.recommenders.ContentRecommender
import ws.vinta.albedo.schemas._
import ws.vinta.albedo.utils.DatasetUtils._

object ContentRecommenderBuilder {
  def main(args: Array[String]): Unit = {
    implicit val spark: SparkSession = SparkSession
      .builder()
      .appName("ContentRecommenderBuilder")
      .getOrCreate()

    import spark.implicits._

    // Load Data

    val rawStarringDS = loadRawStarringDS()

    // Split Data

    val Array(_, testDF) = rawStarringDS.randomSplit(Array(0.9, 0.1))

    val meDF = spark.createDataFrame(Seq(
      (652070, "vinta")
    )).toDF("user_id", "username")

    val testUserDF = testDF
      .select($"user_id")
      .distinct()
      .limit(500)
      .union(meDF.select($"user_id"))
    testUserDF.cache()

    // Make Recommendations

    val topK = 30

    val contentRecommender = new ContentRecommender()
      .setUserCol("user_id")
      .setItemCol("repo_id")
      .setTopK(topK)
      .setEnableEvaluationMode(true)

    val userRecommendedItemDF = contentRecommender.recommendForUsers(testUserDF)
    userRecommendedItemDF.cache()

    userRecommendedItemDF.where($"user_id" === 652070).show(false)

    // Evaluate the Model

    val userActualItemsDS = loadUserActualItemsDF(topK)
      .join(testUserDF, Seq("user_id"))
      .as[UserItems]

    val userPredictedItemsDS = userRecommendedItemDF
      .transform(intoUserPredictedItems($"user_id", $"repo_id", $"score".desc, topK))
      .as[UserItems]

    val rankingEvaluator = new RankingEvaluator(userActualItemsDS)
      .setMetricName("NDCG@k")
      .setK(topK)
      .setUserCol("user_id")
      .setItemsCol("items")
    val metric = rankingEvaluator.evaluate(userPredictedItemsDS)
    println(s"${rankingEvaluator.getFormattedMetricName} = $metric")
    // NDCG@30 = 0.002559563451967487

    spark.stop()
  }
}