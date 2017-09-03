package ws.vinta.albedo

import org.apache.spark.sql.SparkSession
import ws.vinta.albedo.evaluators.RankingEvaluator
import ws.vinta.albedo.evaluators.RankingEvaluator._
import ws.vinta.albedo.utils.DatasetUtils._

object PopularityRecommenderTrainer {
  def main(args: Array[String]): Unit = {
    implicit val spark: SparkSession = SparkSession
      .builder()
      .appName("PopularityRecommenderTrainer")
      .getOrCreate()

    import spark.implicits._

    // Load Data

    val rawUserInfoDS = loadUserInfo()
    rawUserInfoDS.cache()

    val rawRepoInfoDS = loadRepoInfo()
    rawRepoInfoDS.cache()

    val popularReposDS = loadPopularRepos()
    popularReposDS.cache()

    val rawRepoStarringDS = loadRepoStarring()

    // Make Recommendations

    val k = 15

    val userPopularRepoDF = rawUserInfoDS.select($"user_id").crossJoin(popularReposDS.limit(k))

    // Evaluate the Model

    val userActualItemsDF = rawRepoStarringDS.transform(intoUserActualItems($"user_id", $"repo_id", $"starred_at", k))
    val userPredictedItemsDF = userPopularRepoDF.transform(intoUserPredictedItems($"user_id", $"repo_id", $"stargazers_count".desc))

    val rankingEvaluator = new RankingEvaluator(userActualItemsDF)
      .setMetricName("ndcg@k")
      .setK(k)
      .setUserCol("user_id")
      .setItemsCol("items")
    val metric = rankingEvaluator.evaluate(userPredictedItemsDF)
    println(s"${rankingEvaluator.getMetricName} = $metric")

    spark.stop()
  }
}