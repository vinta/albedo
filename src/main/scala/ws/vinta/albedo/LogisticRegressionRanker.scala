package ws.vinta.albedo

import org.apache.spark.SparkConf
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature._
import org.apache.spark.ml.{Pipeline, PipelineModel, Transformer}
import org.apache.spark.ml.recommendation.ALSModel
import org.apache.spark.sql.SparkSession
import ws.vinta.albedo.closures.UDFs._
import ws.vinta.albedo.evaluators.RankingEvaluator
import ws.vinta.albedo.evaluators.RankingEvaluator._
import ws.vinta.albedo.recommenders._
import ws.vinta.albedo.schemas.UserItems
import ws.vinta.albedo.transformers.NegativeBalancer
import ws.vinta.albedo.utils.DatasetUtils._
import ws.vinta.albedo.utils.ModelUtils._

import scala.collection.mutable

object LogisticRegressionRanker {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf()
    conf.set("spark.driver.memory", "4g")
    conf.set("spark.executor.memory", "12g")
    conf.set("spark.executor.cores", "4")

    implicit val spark: SparkSession = SparkSession
      .builder()
      .appName("LogisticRegressionRanker")
      .config(conf)
      .getOrCreate()

    import spark.implicits._

    val sc = spark.sparkContext

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

    // Split Data

    val trainingTestWeights = if (scala.util.Properties.envOrElse("RUN_ON_SMALL_MACHINE", "false") == "true") Array(0.1, 0.9) else Array(0.8, 0.2)
    val takeN = if (scala.util.Properties.envOrElse("RUN_ON_SMALL_MACHINE", "false") == "true") 50 else 500

    val Array(trainingFeaturedDF, testFeaturedDF) = featuredDF.randomSplit(trainingTestWeights)
    trainingFeaturedDF.cache()

    val largeUserIds = testFeaturedDF.select($"user_id").distinct().map(row => row.getInt(0)).collect().toList
    val sampledUserIds = scala.util.Random.shuffle(largeUserIds).take(takeN) :+ 652070
    val testUserDF = spark.createDataFrame(sampledUserIds.map(Tuple1(_))).toDF("user_id")
    testUserDF.cache()

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
      .setMaxIter(10)
      .setRegParam(0.8)
      .setElasticNetParam(0.5)
      .setStandardization(false)
      .setFeaturesCol("features")
      .setLabelCol("starring")

    val pipeline = new Pipeline()
      .setStages((categoricalTransformers :+ alsModel :+ vectorAssembler :+ lr).toArray)

    // Train the Model

    val pipelineModelPath = s"${settings.dataDir}/${settings.today}/rankerPipelineModel.parquet"
    val pipelineModel = loadOrCreateModel[PipelineModel](PipelineModel, pipelineModelPath, () => {
      pipeline.fit(trainingFeaturedDF)
    })

    // Make Recommendations

    val topK = 30

    val alsRecommender = new ALSRecommender()
      .setUserCol("user_id")
      .setItemCol("repo_id")
      .setTopK(topK * 2)

    val contentRecommender = new ContentRecommender()
      .setUserCol("user_id")
      .setItemCol("repo_id")
      .setTopK(topK)
      .setEnableEvaluationMode(true)

    val curationRecommender = new CurationRecommender()
      .setUserCol("user_id")
      .setItemCol("repo_id")
      .setTopK(topK)

    val popularityRecommender = new PopularityRecommender()
      .setUserCol("user_id")
      .setItemCol("repo_id")
      .setTopK(topK)

    val recommenders = mutable.ArrayBuffer.empty[Recommender]
    recommenders += alsRecommender
    //recommenders += contentRecommender
    //recommenders += curationRecommender
    //recommenders += popularityRecommender

    val userRecommendedItemDF = recommenders
      .map((recommender: Transformer) => recommender.transform(testUserDF))
      .reduce(_ union _)
      .select($"user_id", $"repo_id")
      .distinct()

    val userCandidateItemDF = userRecommendedItemDF
      .join(userProfileDF, Seq("user_id"))
      .join(repoProfileDF, Seq("repo_id"))

    // Predict the Ranking

    val userRankedItemDF = pipelineModel.transform(userCandidateItemDF)
    userCandidateItemDF.cache()

    userRankedItemDF
      .where($"user_id" === 652070)
      .select("user_id", "repo_id", "prediction", "probability", "rawPrediction")
      .orderBy(toArrayUDF($"probability").getItem(1).desc)
      .limit(topK)
      .show(false)

    // Evaluate the Model

    val userActualItemsDS = loadUserActualItemsDF(topK)
      .join(testUserDF, Seq("user_id"))
      .as[UserItems]

    val userPredictedItemsDS = userRankedItemDF
      .transform(intoUserPredictedItems($"user_id", $"repo_id", toArrayUDF($"probability").getItem(1).desc, topK))
      .as[UserItems]

    val rankingEvaluator = new RankingEvaluator(userActualItemsDS)
      .setMetricName("NDCG@k")
      .setK(topK)
      .setUserCol("user_id")
      .setItemsCol("items")
    val metric = rankingEvaluator.evaluate(userPredictedItemsDS)
    println(s"${rankingEvaluator.getFormattedMetricName} = $metric")
    // NDCG@30 = 0.0035347470446850824

    spark.stop()
  }
}