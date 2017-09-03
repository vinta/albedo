package ws.vinta.albedo

import org.apache.hadoop.mapred.InvalidInputException
import org.apache.spark.ml.recommendation.{ALS, ALSModel}
import org.apache.spark.sql.SparkSession
import ws.vinta.albedo.evaluators.RankingEvaluator
import ws.vinta.albedo.schemas.UserRecommendations
import ws.vinta.albedo.utils.DatasetUtils.loadRepoStarring
import ws.vinta.albedo.utils.Settings

object ALSRecommenderTrainer {
  def main(args: Array[String]): Unit = {
    implicit val spark: SparkSession = SparkSession
      .builder()
      .getOrCreate()

    import spark.implicits._

    // Load Data

    val rawRepoStarringDS = loadRepoStarring()
    rawRepoStarringDS.cache()
    rawRepoStarringDS.printSchema()

    // Train Model

    val alsModelSavePath = s"${Settings.dataDir}/${Settings.today}/alsModel.parquet"
    val alsModel: ALSModel = try {
      ALSModel.load(alsModelSavePath)
    } catch {
      case e: InvalidInputException => {
        if (e.getMessage().contains("Input path does not exist")) {
          val als = new ALS()
            .setImplicitPrefs(true)
            .setRank(100)
            .setRegParam(0.5)
            .setAlpha(40)
            .setMaxIter(22)
            .setSeed(42)
            .setColdStartStrategy("drop")
            .setUserCol("user_id")
            .setItemCol("repo_id")
            .setRatingCol("starring")
          val alsModel = als.fit(rawRepoStarringDS)
          alsModel.save(alsModelSavePath)
          alsModel
        } else {
          throw e
        }
      }
    }

    // Make Recommendations

    val k = 15

    val userRecommendationsDF = alsModel.recommendForAllUsers(k)
    userRecommendationsDF.printSchema()

    val userRecommendationsDS = userRecommendationsDF.as[UserRecommendations]

    // Evaluate Model

    val userActualItemsDF = RankingEvaluator.createUserActualItems(rawRepoStarringDS, k)
    userActualItemsDF.printSchema()

    val userPredictedItemsDF = userRecommendationsDS.select($"user_id", $"recommendations.repo_id".alias("items"))
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