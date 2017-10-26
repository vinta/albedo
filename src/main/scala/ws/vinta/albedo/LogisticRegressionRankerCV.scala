package ws.vinta.albedo

import org.apache.spark.SparkConf
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer, VectorAssembler}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.recommendation.ALSModel
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.SparkSession
import ws.vinta.albedo.transformers.NegativeBalancer
import ws.vinta.albedo.utils.DatasetUtils._

import scala.collection.mutable

object LogisticRegressionRankerCV {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf()
    if (scala.util.Properties.envOrElse("RUN_ON_SMALL_MACHINE", "false") == "true") {
      conf.setMaster("local[*]")
      conf.set("spark.driver.memory", "12g")
      //conf.setMaster("spark://localhost:7077")
      //conf.set("spark.driver.memory", "2g")
      //conf.set("spark.executor.cores", "3")
      //conf.set("spark.executor.memory", "12g")
      //conf.setJars(List("target/albedo-1.0.0-SNAPSHOT-uber.jar"))
    }

    implicit val spark: SparkSession = SparkSession
      .builder()
      .appName("LogisticRegressionRankerCV")
      .config(conf)
      .getOrCreate()

    import spark.implicits._

    val sc = spark.sparkContext
    sc.setCheckpointDir("./spark-data/checkpoint")

    // Load Data

    val userProfileDF = loadUserProfileDF().select($"user_id", $"features".alias("user_features"))

    val repoProfileDF = loadRepoProfileDF().select($"repo_id", $"features".alias("repo_features"))

    val rawStarringDS = loadRawStarringDS()

    // Handle Imbalanced Samples

    val featuredDFpath = s"${settings.dataDir}/${settings.today}/featuredDF.parquet"
    val featuredDF = loadOrCreateDataFrame(featuredDFpath, () => {
      val popularReposDS = loadPopularRepoDF()
      val popularRepos = popularReposDS
        .select($"repo_id".as[Int])
        .collect()
        .to[mutable.LinkedHashSet]
      val bcPopularRepos = sc.broadcast(popularRepos)

      val negativeBalancer = new NegativeBalancer(bcPopularRepos)
        .setUserCol("user_id")
        .setItemCol("repo_id")
        .setLabelCol("starring")
        .setNegativeValue(0.0)
        .setNegativePositiveRatio(1.0)
      val balancedStarringDF = negativeBalancer.transform(rawStarringDS)

      balancedStarringDF
        .join(userProfileDF, Seq("user_id"))
        .join(repoProfileDF, Seq("repo_id"))
    })

    // Build the Model Pipeline

    val categoricalColumnNames = mutable.ArrayBuffer("user_id", "repo_id")
    val categoricalTransformers = categoricalColumnNames.flatMap((columnName: String) => {
      val stringIndexer = new StringIndexer()
        .setInputCol(columnName)
        .setOutputCol(s"${columnName}_idx")
        .setHandleInvalid("keep")

      val oneHotEncoder = new OneHotEncoder()
        .setInputCol(s"${columnName}_idx")
        .setOutputCol(s"${columnName}_ohe")
        .setDropLast(true)

      Array(stringIndexer, oneHotEncoder)
    })

    val alsModelPath = s"${settings.dataDir}/${settings.today}/alsModel.parquet"
    val alsModel = ALSModel.load(alsModelPath)
      .setUserCol("user_id")
      .setItemCol("repo_id")
      .setPredictionCol("als_score")
      .setColdStartStrategy("drop")

    val vectorAssembler = new VectorAssembler()
      .setInputCols(Array("user_id_ohe", "repo_id_ohe", "user_features", "repo_features", "als_score"))
      .setOutputCol("features")

    val lr = new LogisticRegression()
      .setFeaturesCol("features")
      .setLabelCol("starring")

    val pipeline = new Pipeline()
      .setStages((categoricalTransformers :+ alsModel :+ vectorAssembler :+ lr).toArray)

    // Cross-validate Models

    val subsetFeaturedDF = featuredDF
      .sample(withReplacement = true, 0.3)
      .cache()

    val paramGrid = new ParamGridBuilder()
      .addGrid(lr.maxIter, Array(10, 100))
      .addGrid(lr.regParam, Array(0.5, 0.8))
      .addGrid(lr.elasticNetParam, Array(0.05, 0.2))
      .addGrid(lr.threshold, Array(0.25, 0.5, 0.75))
      .addGrid(lr.standardization, Array(false, true))
      .build()

    val evaluator = new BinaryClassificationEvaluator()
      .setMetricName("areaUnderROC")
      .setRawPredictionCol("rawPrediction")
      .setLabelCol("starring")

    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEstimatorParamMaps(paramGrid)
      .setEvaluator(evaluator)
      .setNumFolds(2)

    val cvModel = cv.fit(subsetFeaturedDF)

    // Show Best Parameters

    cvModel.getEstimatorParamMaps
      .zip(cvModel.avgMetrics)
      .sortWith(_._2 > _._2)
      .foreach((pair: (ParamMap, Double)) => {
        println(s"${pair._2}: ${pair._1}")
      })

    spark.stop()
  }
}