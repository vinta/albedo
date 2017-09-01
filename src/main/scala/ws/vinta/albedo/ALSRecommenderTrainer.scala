package ws.vinta.albedo

import org.apache.spark.ml.recommendation.ALS
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.collect_list
import ws.vinta.albedo.evaluators.RankingEvaluator
import ws.vinta.albedo.schemas.UserRecommendations
import ws.vinta.albedo.utils.DataSourceUtils.loadRepoStarring
import ws.vinta.albedo.utils.Settings

object ALSRecommenderTrainer {
  def main(args: Array[String]): Unit = {
    implicit val spark: SparkSession = SparkSession
      .builder()
      .appName("ALSRecommenderTrainer")
      .getOrCreate()

    import spark.implicits._

    val sc = spark.sparkContext

    val rawRepoStarringDS = loadRepoStarring()
    rawRepoStarringDS.cache()
    rawRepoStarringDS.printSchema()

    val als = new ALS()
      .setImplicitPrefs(true)
      .setRank(50)
      .setRegParam(0.5)
      .setAlpha(40)
      .setMaxIter(20)
      .setSeed(42)
      .setColdStartStrategy("drop")
      .setUserCol("user_id")
      .setItemCol("repo_id")
      .setRatingCol("starring")
    val alsModel = als.fit(rawRepoStarringDS)
    alsModel.save(s"${Settings.dataDir}/${Settings.today}/alsModel.parquet")

    val k = 15

    val userRecommendationsDF = alsModel.recommendForAllUsers(k)
    userRecommendationsDF.printSchema()

    val userRecommendationsDS = userRecommendationsDF.as[UserRecommendations]

    val userActualItemsDF = rawRepoStarringDS
      .groupBy($"user_id")
      .agg(collect_list($"repo_id").alias("items"))
    val rankingEvaluator = new RankingEvaluator(userActualItemsDF)
      .setMetricName("ndcg@k")
      .setK(k)
    val metric = rankingEvaluator.evaluate(userRecommendationsDS)
    println(s"${rankingEvaluator.getMetricName} = $metric")

    spark.stop()
  }
}