package ws.vinta.albedo

import org.apache.spark.sql.SparkSession
import ws.vinta.albedo.evaluators.RankingEvaluator
import ws.vinta.albedo.evaluators.RankingEvaluator._
import ws.vinta.albedo.schemas._
import ws.vinta.albedo.utils.DatasetUtils._

object PopularityRecommender {
  def main(args: Array[String]): Unit = {
    implicit val spark: SparkSession = SparkSession
      .builder()
      .appName("PopularityRecommender")
      .getOrCreate()

    import spark.implicits._

    // Load Data

    val rawStarringDS = loadRawStarringDS()
    rawStarringDS.cache()

    val rawUserInfoDS = loadRawUserInfoDS()
    rawUserInfoDS.cache()

    // Split Data

    // 統一使用 20% 的 test set 來評估每個演算法
    val Array(_, testDF) = rawStarringDS.randomSplit(Array(0.8, 0.2))
    testDF.cache()

    val testUserDF = testDF.select($"user_id").distinct()

    // Make Recommendations

    val topK = 30

    val popularRepoDF = loadPopularRepoDF().limit(topK)
    popularRepoDF.cache()

    val userPopularRepoDF = rawUserInfoDS
      .select($"user_id")
      .crossJoin(popularRepoDF)

    // Evaluate the Model

    val userActualItemsDS = testDF
      .transform(intoUserActualItems($"user_id", $"repo_id", $"starred_at".desc, topK))
      .as[UserItems]

    val userPredictedItemsDS = userPopularRepoDF
      .join(testUserDF, Seq("user_id"))
      .transform(intoUserPredictedItems($"user_id", $"repo_id", $"stargazers_count".desc, topK))
      .as[UserItems]

    val rankingEvaluator = new RankingEvaluator(userActualItemsDS)
      .setMetricName("ndcg@k")
      .setK(topK)
      .setUserCol("user_id")
      .setItemsCol("items")
    val metric = rankingEvaluator.evaluate(userPredictedItemsDS)
    println(s"${rankingEvaluator.getMetricName} = $metric")
    // NDCG@k = 0.0013035714524256231

    spark.stop()
  }
}
