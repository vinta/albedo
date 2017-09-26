package ws.vinta.albedo

import org.apache.hadoop.mapred.InvalidInputException
import org.apache.spark.ml.recommendation.{ALS, ALSModel}
import org.apache.spark.sql.SparkSession
import ws.vinta.albedo.evaluators.RankingEvaluator
import ws.vinta.albedo.evaluators.RankingEvaluator._
import ws.vinta.albedo.schemas.{UserItems, UserRecommendations}
import ws.vinta.albedo.utils.DatasetUtils._

object ALSRecommender {
  def main(args: Array[String]): Unit = {
    implicit val spark: SparkSession = SparkSession
      .builder()
      .appName("ALSRecommenderTrainer")
      .getOrCreate()

    import spark.implicits._

    // Load Data

    val rawRepoStarringDS = loadRepoStarring()
    rawRepoStarringDS.cache()

    // Train the Model

    val alsModelSavePath = s"${settings.dataDir}/${settings.today}/alsModel.parquet"
    val alsModel: ALSModel = try {
      ALSModel.load(alsModelSavePath)
    } catch {
      case e: InvalidInputException => {
        if (e.getMessage().contains("Input path does not exist")) {
          val als = new ALS()
            .setImplicitPrefs(true)
            .setRank(50)
            .setRegParam(0.5)
            .setAlpha(40)
            .setMaxIter(25)
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

    val userRecommendationsDS = alsModel
      .recommendForAllUsers(k)
      .as[UserRecommendations]

    // Evaluate the Model

    val userActualItemsDS = rawRepoStarringDS
      .transform(intoUserActualItems($"user_id", $"repo_id", $"starred_at".desc, k))
      .as[UserItems]

    val userPredictedItemsDS = userRecommendationsDS
      .transform(intoUserPredictedItems($"user_id", $"recommendations.repo_id"))
      .as[UserItems]

    val rankingEvaluator = new RankingEvaluator(userActualItemsDS)
      .setMetricName("ndcg@k")
      .setK(k)
      .setUserCol("user_id")
      .setItemsCol("items")
    val metric = rankingEvaluator.evaluate(userPredictedItemsDS)
    println(s"${rankingEvaluator.getMetricName} = $metric")

    spark.stop()
  }
}