package ws.vinta.albedo

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.collect_list
import ws.vinta.albedo.evaluators.RankingEvaluator
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

    val userItemDF = rawUserInfoDS.select($"user_id").crossJoin(popularReposDS.limit(k))
    userItemDF.printSchema()

    // Evaluate the Model

    val userActualItemsDF = RankingEvaluator.createUserActualItems(rawRepoStarringDS, k)
    userActualItemsDF.printSchema()

    val userPredictedItemsDF = userItemDF
      .orderBy($"stargazers_count".desc)
      .groupBy($"user_id")
      .agg(collect_list($"repo_id").alias("items"))
    userPredictedItemsDF.printSchema()

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