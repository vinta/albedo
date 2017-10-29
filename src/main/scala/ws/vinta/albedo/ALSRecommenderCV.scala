package ws.vinta.albedo

import org.apache.spark.SparkConf
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.recommendation.ALS
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.SparkSession
import ws.vinta.albedo.evaluators.RankingEvaluator
import ws.vinta.albedo.evaluators.RankingEvaluator._
import ws.vinta.albedo.schemas.UserItems
import ws.vinta.albedo.transformers.RankingMetricFormatterALS
import ws.vinta.albedo.utils.DatasetUtils._

object ALSRecommenderCV {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf()
    if (scala.util.Properties.envOrElse("RUN_WITH_INTELLIJ", "false") == "true") {
      conf.setMaster("local[*]")
      conf.set("spark.driver.memory", "12g")
      //conf.setMaster("local-cluster[1, 3, 12288]")
      //conf.setJars(List("target/albedo-1.0.0-SNAPSHOT-uber.jar"))
      //conf.setMaster("spark://localhost:7077")
      //conf.set("spark.driver.memory", "2g")
      //conf.set("spark.executor.cores", "3")
      //conf.set("spark.executor.memory", "12g")
      //conf.setJars(List("target/albedo-1.0.0-SNAPSHOT-uber.jar"))
    }

    implicit val spark: SparkSession = SparkSession
      .builder()
      .appName("ALSRecommenderCV")
      .config(conf)
      .getOrCreate()

    import spark.implicits._

    val sc = spark.sparkContext
    sc.setCheckpointDir("./spark-data/checkpoint")

    // Load Data

    val rawStarringDS = loadRawStarringDS().cache()

    // Build the Pipeline

    val als = new ALS()
      .setImplicitPrefs(true)
      .setSeed(42)
      .setColdStartStrategy("drop")
      .setUserCol("user_id")
      .setItemCol("repo_id")
      .setRatingCol("starring")

    val alsPredictionFormatter = new RankingMetricFormatterALS()
      .setUserCol("user_id")
      .setItemCol("repo_id")
      .setPredictionCol("prediction")

    val pipeline = new Pipeline()
      .setStages(Array(als, alsPredictionFormatter))

    // Cross-validate Models

    val paramGrid = new ParamGridBuilder()
      .addGrid(als.rank, Array(50, 100))
      .addGrid(als.regParam, Array(0.01, 0.1, 1.0))
      .addGrid(als.alpha, Array(0.01, 0.1, 40))
      .addGrid(als.maxIter, Array(25))
      .build()

    val topK = 30

    val userActualItemsDS = loadUserActualItemsDF(topK)
      .as[UserItems]
      .cache()

    val evaluator = new RankingEvaluator(userActualItemsDS)
      .setMetricName("NDCG@k")
      .setK(topK)
      .setUserCol("user_id")
      .setItemsCol("items")

    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEstimatorParamMaps(paramGrid)
      .setEvaluator(evaluator)
      .setNumFolds(2)

    val cvModel = cv.fit(rawStarringDS)

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