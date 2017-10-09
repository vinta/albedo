package ws.vinta.albedo

import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import ws.vinta.albedo.evaluators.RankingEvaluator
import ws.vinta.albedo.evaluators.RankingEvaluator._
import ws.vinta.albedo.recommenders.ContentRecommender
import ws.vinta.albedo.schemas._
import ws.vinta.albedo.utils.DatasetUtils._

object ContentRecommenderBuilder {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf()
    conf.set("spark.driver.memory", "4g")
    conf.set("spark.executor.memory", "12g")
    conf.set("spark.executor.cores", "4")

    implicit val spark: SparkSession = SparkSession
      .builder()
      .appName("ContentRecommenderBuilder")
      .config(conf)
      .getOrCreate()

    import spark.implicits._

    val sc = spark.sparkContext
    sc.setCheckpointDir("./spark-data/checkpoint")

    // Load Data

    val rawRepoInfoDS = loadRawRepoInfoDS()

    val rawStarringDS = loadRawStarringDS()

    // Split Data

    // 雖然不是每個演算法都需要劃分 training set 和 test set
    // 不過為了方便比較，我們還是統一使用 20% 的 test set 來評估每個模型
    val Array(_, testDF) = rawStarringDS.randomSplit(Array(0.8, 0.2))
    testDF.cache()

    val meDF = spark.createDataFrame(Seq(
      (652070, "vinta")
    )).toDF("user_id", "username")

    val testUserDF = testDF.select($"user_id").union(meDF.select($"user_id")).distinct().limit(5)

    // Make Recommendations

    val topK = 30

    val contentRecommender = new ContentRecommender()
      .setUserCol("user_id")
      .setItemCol("repo_id")
      .setTopK(topK)

    val userRecommendedItemDF = contentRecommender.recommendForUsers(testUserDF)
    userRecommendedItemDF.cache()

    userRecommendedItemDF.where($"user_id" === 652070).show(false)

    // Evaluate the Model

    //val userActualItemsDS = testDF
    //  .transform(intoUserActualItems($"user_id", $"repo_id", $"starred_at".desc, topK))
    //  .as[UserItems]
    //
    //val userPredictedItemsDS = userRecommendedItemDF
    //  .join(testUserDF, Seq("user_id"))
    //  .transform(intoUserPredictedItems($"user_id", $"repo_id", $"distance".asc, topK))
    //  .as[UserItems]
    //
    //val rankingEvaluator = new RankingEvaluator(userActualItemsDS)
    //  .setMetricName("ndcg@k")
    //  .setK(topK)
    //  .setUserCol("user_id")
    //  .setItemsCol("items")
    //val metric = rankingEvaluator.evaluate(userPredictedItemsDS)
    //println(s"${rankingEvaluator.getMetricName} = $metric")
    //// NDCG@k = ???

    spark.stop()
  }
}