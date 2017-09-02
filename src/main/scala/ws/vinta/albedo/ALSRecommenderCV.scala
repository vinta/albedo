package ws.vinta.albedo

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.recommendation.ALS
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.collect_list
import ws.vinta.albedo.evaluators.RankingEvaluator
import ws.vinta.albedo.preprocessors.RecommendationFormatter
import ws.vinta.albedo.utils.DataSourceUtils.loadRepoStarring

object ALSRecommenderCV {
  def main(args: Array[String]): Unit = {
    implicit val spark: SparkSession = SparkSession
      .builder()
      .appName("ALSRecommenderCV")
      .getOrCreate()

    import spark.implicits._

    // Load Data

    val rawRepoStarringDS = loadRepoStarring()
    rawRepoStarringDS.cache()
    rawRepoStarringDS.printSchema()

    // Cross-validate Model

    val k = 15

    val als = new ALS()
      .setImplicitPrefs(true)
      .setSeed(42)
      .setColdStartStrategy("drop")
      .setUserCol("user_id")
      .setItemCol("repo_id")
      .setRatingCol("starring")

    val recommendationFormatter = new RecommendationFormatter()
      .setUserCol("user_id")
      .setItemCol("repo_id")
      .setPredictionCol("prediction")
      .setOutputCol("recommendations")

    val pipeline = new Pipeline()
      .setStages(Array(als, recommendationFormatter))

    val paramGrid = new ParamGridBuilder()
      .addGrid(als.rank, Array(50, 100, 200))
      .addGrid(als.regParam, Array(0.01, 0.1, 0.5))
      .addGrid(als.alpha, Array(0.01, 0.5, 1, 40))
      .addGrid(als.maxIter, Array(25))
      .build()

    val userActualItemsDF = rawRepoStarringDS
      .orderBy($"starred_at".desc)
      .groupBy($"user_id")
      .agg(collect_list($"repo_id").alias("items"))

    val rankingEvaluator = new RankingEvaluator(userActualItemsDF)
      .setMetricName("ndcg@k")
      .setK(k)

    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEstimatorParamMaps(paramGrid)
      .setEvaluator(rankingEvaluator)
      .setNumFolds(2)

    val cvModel = cv.fit(rawRepoStarringDS)

    // Show Best Parameters

    cvModel.getEstimatorParamMaps
      .zip(cvModel.avgMetrics)
      .sortWith(_._2 > _._2) // (paramMaps, metric)
      .foreach((pair: (ParamMap, Double)) => {
        println(s"${pair._2}: ${pair._1}")
      })

    spark.stop()
  }
}