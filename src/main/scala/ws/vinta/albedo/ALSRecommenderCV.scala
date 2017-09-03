package ws.vinta.albedo

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.recommendation.ALS
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.SparkSession
import ws.vinta.albedo.evaluators.RankingEvaluator
import ws.vinta.albedo.evaluators.RankingEvaluator.intoUserActualItems
import ws.vinta.albedo.preprocessors.PredictionFormatter
import ws.vinta.albedo.utils.DatasetUtils._
import ws.vinta.albedo.utils.Settings

object ALSRecommenderCV {
  def main(args: Array[String]): Unit = {
    implicit val spark: SparkSession = SparkSession
      .builder()
      .appName("ALSRecommenderCV")
      .getOrCreate()

    import spark.implicits._

    val sc = spark.sparkContext
    sc.setCheckpointDir(s"${Settings.dataDir}/checkpoint")

    // Load Data

    val rawRepoStarringDS = loadRepoStarring()
    rawRepoStarringDS.cache()

    // Build the Pipeline

    val als = new ALS()
      .setImplicitPrefs(true)
      .setSeed(42)
      .setColdStartStrategy("drop")
      .setUserCol("user_id")
      .setItemCol("repo_id")
      .setRatingCol("starring")

    val predictionFormatter = new PredictionFormatter()
      .setUserCol("user_id")
      .setItemCol("repo_id")
      .setPredictionCol("prediction")

    val pipeline = new Pipeline()
      .setStages(Array(als, predictionFormatter))

    // Cross-validate Models

    val paramGrid = new ParamGridBuilder()
      .addGrid(als.rank, Array(50, 100))
      .addGrid(als.regParam, Array(0.01, 0.1, 1.0))
      .addGrid(als.alpha, Array(0.01, 0.1, 40))
      .addGrid(als.maxIter, Array(25))
      .build()

    val k = 15

    val userActualItemsDF = rawRepoStarringDS.transform(intoUserActualItems($"user_id", $"repo_id", $"starred_at", k))
    userActualItemsDF.cache()

    val rankingEvaluator = new RankingEvaluator(userActualItemsDF)
      .setMetricName("ndcg@k")
      .setK(k)
      .setUserCol("user_id")
      .setItemsCol("items")

    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEstimatorParamMaps(paramGrid)
      .setEvaluator(rankingEvaluator)
      .setNumFolds(2)

    val cvModel = cv.fit(rawRepoStarringDS)

    // Show Best Parameters

    cvModel.getEstimatorParamMaps
      .zip(cvModel.avgMetrics) // (paramMaps, metric)
      .sortWith(_._2 > _._2) // _._2 就是 metric
      .foreach((pair: (ParamMap, Double)) => {
        println(s"${pair._2}: ${pair._1}")
      })

    spark.stop()
  }
}