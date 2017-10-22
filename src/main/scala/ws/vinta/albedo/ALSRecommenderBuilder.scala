package ws.vinta.albedo

import org.apache.spark.SparkConf
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
    val conf = new SparkConf()
    if (scala.util.Properties.envOrElse("RUN_ON_SMALL_MACHINE", "false") == "true") {
      conf.set("spark.driver.memory", "4g")
      conf.set("spark.executor.memory", "12g")
      conf.set("spark.executor.cores", "4")
    }

    implicit val spark: SparkSession = SparkSession
      .builder()
      .appName("ALSRecommenderBuilder")
      .config(conf)
      .getOrCreate()

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

      als.fit(rawStarringDS)
    })

    // Split Data

    val Array(_, testDF) = rawStarringDS.randomSplit(Array(0.9, 0.1))

    val largeUserIds = testDF.select($"user_id").distinct().map(row => row.getInt(0)).collect().toList
    val sampledUserIds = scala.util.Random.shuffle(largeUserIds).take(500) :+ 652070
    val testUserDF = spark.createDataFrame(sampledUserIds.map(Tuple1(_))).toDF("user_id")
    testUserDF.cache()

    // Make Recommendations

    val topK = 30

    val alsRecommender = new ALSRecommender()
      .setUserCol("user_id")
      .setItemCol("repo_id")
      .setTopK(topK)

    val userRecommendedItemDF = alsRecommender.recommendForUsers(testUserDF)
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
    // NDCG@30 = 0.04818769346035656

    spark.stop()
  }
}