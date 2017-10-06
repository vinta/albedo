package ws.vinta.albedo

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import ws.vinta.albedo.evaluators.RankingEvaluator
import ws.vinta.albedo.evaluators.RankingEvaluator._
import ws.vinta.albedo.schemas._
import ws.vinta.albedo.utils.DatasetUtils._

object CurationRecommenderBuilder {
  def main(args: Array[String]): Unit = {
    implicit val spark: SparkSession = SparkSession
      .builder()
      .appName("CurationRecommender")
      .getOrCreate()

    import spark.implicits._

    // Load Data

    val rawStarringDS = loadRawStarringDS()
    rawStarringDS.cache()

    val rawUserInfoDS = loadRawUserInfoDS()
    rawUserInfoDS.cache()

    // 統一使用 20% 的 test set 來評估每個演算法
    val Array(_, testDF) = rawStarringDS.randomSplit(Array(0.8, 0.2))
    testDF.cache()

    val testUserDF = testDF.select($"user_id").distinct()

    // Make Recommendations

    val topK = 30

    val curatorIds = Array(652070, 1912583, 59990, 646843, 28702) // vinta, saiday, tzangms, fukuball, wancw
    val curatedRepoDF = rawStarringDS
      .select($"repo_id", $"starred_at")
      .where($"user_id".isin(curatorIds: _*))
      .groupBy($"repo_id")
      .agg(max($"starred_at").alias("starred_at"))
      .orderBy($"starred_at".desc)
      .limit(topK)
    curatedRepoDF.cache()

    val userCuratedRepoDF = rawUserInfoDS
      .select($"user_id")
      .crossJoin(curatedRepoDF)

    // Evaluate the Model

    val userActualItemsDS = testDF
      .transform(intoUserActualItems($"user_id", $"repo_id", $"starred_at".desc, topK))
      .as[UserItems]

    val userPredictedItemsDS = userCuratedRepoDF
      .join(testUserDF, Seq("user_id"))
      .transform(intoUserPredictedItems($"user_id", $"repo_id", $"starred_at".desc, topK))
      .as[UserItems]

    val rankingEvaluator = new RankingEvaluator(userActualItemsDS)
      .setMetricName("ndcg@k")
      .setK(topK)
      .setUserCol("user_id")
      .setItemsCol("items")
    val metric = rankingEvaluator.evaluate(userPredictedItemsDS)
    println(s"${rankingEvaluator.getMetricName} = $metric")
    // NDCG@k = 0.000259959

    spark.stop()
  }
}
