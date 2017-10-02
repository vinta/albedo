package ws.vinta.albedo

import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature._
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.SparkSession
import ws.vinta.albedo.closures.UDFs._
import ws.vinta.albedo.evaluators.RankingEvaluator
import ws.vinta.albedo.evaluators.RankingEvaluator._
import ws.vinta.albedo.transformers.NegativeBalancer
import ws.vinta.albedo.utils.DatasetUtils._
import ws.vinta.albedo.utils.ModelUtils._

import scala.collection.mutable

object LogisticRegressionRanker {
  def main(args: Array[String]): Unit = {
    implicit val spark: SparkSession = SparkSession
      .builder()
      .appName("LogisticRegressionRanker")
      .getOrCreate()

    import spark.implicits._

    val sc = spark.sparkContext
    sc.setCheckpointDir("./spark-data/checkpoint")

    // Load Data

    val userProfileDF = loadUserProfileDF().select($"user_id", $"features".alias("user_features"))
    userProfileDF.cache()

    val repoProfileDF = loadRepoProfileDF().select($"repo_id", $"features".alias("repo_features"))
    repoProfileDF.cache()

    val rawStarringDS = loadRawStarringDS()
    rawStarringDS.cache()

    // Handle Imbalanced Samples

    val fullDFpath = s"${settings.dataDir}/${settings.today}/fullDF.parquet"
    val fullDF = loadOrCreateDataFrame(fullDFpath, () => {
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

      val fullDF = balancedStarringDF
        .join(userProfileDF, Seq("user_id"))
        .join(repoProfileDF, Seq("repo_id"))
      fullDF
    })

    // Split Data

    val Array(trainingDF, testDF) = fullDF.randomSplit(Array(0.8, 0.2))
    trainingDF.cache()
    testDF.cache()

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

    val vectorAssembler = new VectorAssembler()
      .setInputCols(Array("user_id_ohe", "repo_id_ohe", "user_features", "repo_features"))
      .setOutputCol("features")

    val lr = new LogisticRegression()
      .setMaxIter(10)
      .setRegParam(0.0)
      .setElasticNetParam(0.0)
      .setStandardization(true)
      .setFeaturesCol("features")
      .setLabelCol("starring")

    val pipeline: Pipeline = new Pipeline()
      .setStages((categoricalTransformers :+ vectorAssembler :+ lr).toArray)

    // Train the Model

    val pipelineModelPath = s"${settings.dataDir}/${settings.today}/rankerPipelineModel.parquet"
    val pipelineModel = loadOrCreateModel[PipelineModel](PipelineModel, pipelineModelPath, () => {
      pipeline.fit(trainingDF)
    })

    // Make Recommendations

    // TODO:
    // columns: user_id, repo_id, popular_score, als_score, word2vec_score
    // item candidates from Popular
    // item candidates from ALS
    // item candidates from Word2Vec

    val condidateDF = spark.createDataFrame(Seq(
      (1, 1, 0.1),
      (1, 2, 1.0),
      (1, 3, 0.8)
    ))

    // Predict the Ranking

    val resultTestDF = pipelineModel.transform(testDF)

    resultTestDF
      .select("user_id", "repo_id", "starring", "prediction", "probability")
      .where("user_id = 652070")
      .orderBy(toArrayUDF($"probability").getItem(1).desc)
      .show(false)

    // Evaluate the Model

    val topK = 30

    val userActualItemsDF = resultTestDF
      .where($"starring" === 1.0)
      .join(rawStarringDS, Seq("user_id", "repo_id", "starring"), "left_outer") // 為了找回 starred_at 欄位
      .transform(intoUserActualItems($"user_id", $"repo_id", $"starred_at".desc, topK))

    val userPredictedItemsDF = resultTestDF.transform(intoUserPredictedItems($"user_id", $"repo_id", toArrayUDF($"probability").getItem(1).desc, topK))

    val rankingEvaluator = new RankingEvaluator(userActualItemsDF)
      .setMetricName("ndcg@k")
      .setK(topK)
      .setUserCol("user_id")
      .setItemsCol("items")
    val metric = rankingEvaluator.evaluate(userPredictedItemsDF)
    println(s"${rankingEvaluator.getMetricName} = $metric")
    // NDCG@k = ???

    spark.stop()
  }
}
