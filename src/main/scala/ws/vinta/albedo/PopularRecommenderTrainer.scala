package ws.vinta.albedo

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.collect_list
import ws.vinta.albedo.evaluators.RankingEvaluator
import ws.vinta.albedo.preprocessors.popularReposBuilder
import ws.vinta.albedo.utils.DataSourceUtils.loadRepoInfo
object PopularRecommenderTrainer {
  def main(args: Array[String]): Unit = {
    implicit val spark: SparkSession = SparkSession
      .builder()
      .appName("PopularRecommenderTrainer")
      .getOrCreate()

    import spark.implicits._

    val sc = spark.sparkContext

    // Load Data

    val rawRepoInfoDS = loadRepoInfo()
    rawRepoInfoDS.cache()

    val popularReposDF = popularReposBuilder.transform(rawRepoInfoDS)

    // Make Recommendations

    val k = 15

    //val userPredictedItemsDF = null;
    //
    //val userActualItemsDF = rawRepoStarringDS
    //  .orderBy($"starred_at".desc)
    //  .groupBy($"user_id")
    //  .agg(collect_list($"repo_id").alias("items"))
    //
    //val rankingEvaluator = new RankingEvaluator(userActualItemsDF)
    //  .setMetricName("ndcg@k")
    //  .setK(k)
    //val metric = rankingEvaluator.evaluate(userRecommendationsDS)
    //println(s"${rankingEvaluator.getMetricName} = $metric")

    spark.stop()
  }
}