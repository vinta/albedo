package ws.vinta.albedo

import org.apache.spark.ml.recommendation.{ALS, ALSModel}
import org.apache.spark.sql.SparkSession
import ws.vinta.albedo.evaluators.RankingEvaluator
import ws.vinta.albedo.evaluators.RankingEvaluator._
import ws.vinta.albedo.schemas._
import ws.vinta.albedo.utils.DatasetUtils._
import ws.vinta.albedo.utils.ModelUtils._
import ws.vinta.albedo.recommenders.ALSRecommender

object ALSRecommenderBuilder {
  def main(args: Array[String]): Unit = {
    implicit val spark: SparkSession = SparkSession
      .builder()
      .appName("ALSRecommender")
      .getOrCreate()

    val sc = spark.sparkContext
    sc.setCheckpointDir("./spark-data/checkpoint")

    import spark.implicits._

    // Load Data

    val rawStarringDS = loadRawStarringDS()
    rawStarringDS.cache()

    // Train the Model

    val alsModelPath = s"${settings.dataDir}/${settings.today}/alsModel.parquet"
    val alsModel = loadOrCreateModel[ALSModel](ALSModel, alsModelPath, () => {
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

      // 跟其他演算法不同，我們是用整個 dataset 來訓練 ALS
      //
      val alsModel = als.fit(rawStarringDS)
      alsModel
    })

    // Split Data

    // 雖然不是每個演算法都需要劃分 training set 和 test set
    // 不過為了方便比較，我們還是統一使用 20% 的 test set 來評估每個模型
    val Array(_, testDF) = rawStarringDS.randomSplit(Array(0.8, 0.2))
    testDF.cache()

    val testUserDF = testDF.select($"user_id").distinct()

    // Make Recommendations

    val topK = 30

    val alsRecommender = new ALSRecommender()
      .setUserCol("user_id")
      .setItemCol("repo_id")
      .setTopK(topK)

    val userRecommendedItemDF = alsRecommender.recommendForUsers(testUserDF)

    // Evaluate the Model

    val userActualItemsDS = testDF
      .transform(intoUserActualItems($"user_id", $"repo_id", $"starred_at".desc, topK))
      .as[UserItems]

    val userPredictedItemsDS = userRecommendedItemDF
      .transform(intoUserPredictedItems($"user_id", $"repo_id", $"score".desc, topK))
      .as[UserItems]

    val rankingEvaluator = new RankingEvaluator(userActualItemsDS)
      .setMetricName("ndcg@k")
      .setK(topK)
      .setUserCol("user_id")
      .setItemsCol("items")
    val metric = rankingEvaluator.evaluate(userPredictedItemsDS)
    println(s"${rankingEvaluator.getMetricName} = $metric")
    // NDCG@k = 0.05026158143766048

    spark.stop()
  }
}