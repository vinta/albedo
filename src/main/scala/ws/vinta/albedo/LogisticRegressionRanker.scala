package ws.vinta.albedo

import org.apache.spark.SparkConf
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature._
import org.apache.spark.ml.recommendation.ALSModel
import org.apache.spark.ml.{Pipeline, PipelineModel, PipelineStage}
import org.apache.spark.sql.SparkSession
import ws.vinta.albedo.closures.UDFs._
import ws.vinta.albedo.evaluators.RankingEvaluator
import ws.vinta.albedo.evaluators.RankingEvaluator._
import ws.vinta.albedo.recommenders._
import ws.vinta.albedo.schemas.UserItems
import ws.vinta.albedo.transformers._
import ws.vinta.albedo.utils.DatasetUtils._
import ws.vinta.albedo.utils.ModelUtils._

import scala.collection.mutable

object LogisticRegressionRanker {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf()
    if (scala.util.Properties.envOrElse("RUN_ON_SMALL_MACHINE", "false") == "true") {
      conf.set("spark.driver.memory", "4g")
      conf.set("spark.executor.memory", "12g")
      conf.set("spark.executor.cores", "4")
    }

    implicit val spark: SparkSession = SparkSession
      .builder()
      .appName("LogisticRegressionRanker")
      .config(conf)
      .getOrCreate()

    import spark.implicits._

    val sc = spark.sparkContext

    // Load Data

    val userProfileDF = loadUserProfileDF()

    val repoProfileDF = loadRepoProfileDF()

    val rawStarringDS = loadRawStarringDS()

    // Handle Imbalanced Samples

    val balancedStarringDFpath = s"${settings.dataDir}/${settings.today}/balancedStarringDF.parquet"
    val balancedStarringDF = loadOrCreateDataFrame(balancedStarringDFpath, () => {
      val popularReposDS = loadPopularRepoDF()
      val popularRepos = popularReposDS
        .select($"repo_id".as[Int])
        .collect()
        .to[mutable.LinkedHashSet]
      val bcPopularRepos = sc.broadcast(popularRepos)

      val negativeBalancer = new NegativeBalancer(bcPopularRepos)
        .setUserCol("user_id")
        .setItemCol("repo_id")
        .setTimeCol("starred_at")
        .setLabelCol("starring")
        .setNegativeValue(0.0)
        .setNegativePositiveRatio(1.0)
      negativeBalancer.transform(rawStarringDS)
    })
    .repartition($"user_id")

    // Feature Engineering

    val continuousColumnNames = mutable.ArrayBuffer.empty[String]
    val categoricalColumnNames = mutable.ArrayBuffer.empty[String]
    val listColumnNames = mutable.ArrayBuffer.empty[String]
    val textColumnNames = mutable.ArrayBuffer.empty[String]

    val featuredDF = balancedStarringDF
      .join(userProfileDF, Seq("user_id"))
      .join(repoProfileDF, Seq("repo_id"))
      .repartition($"user_id")
      .cache()

    categoricalColumnNames += "user_id"
    categoricalColumnNames += "repo_id"

    // User Profile

    continuousColumnNames += "user_public_repos_count"
    continuousColumnNames += "user_public_gists_count"
    continuousColumnNames += "user_followers_count"
    continuousColumnNames += "user_following_count"
    continuousColumnNames += "user_followers_following_ratio"
    continuousColumnNames += "user_days_between_created_at_today"
    continuousColumnNames += "user_days_between_updated_at_today"
    continuousColumnNames += "user_starred_repos_count"
    continuousColumnNames += "user_avg_daily_starred_repos_count"

    categoricalColumnNames += "user_account_type"
    categoricalColumnNames += "user_has_null"
    categoricalColumnNames += "user_knows_web"
    categoricalColumnNames += "user_knows_backend"
    categoricalColumnNames += "user_knows_frontend"
    categoricalColumnNames += "user_knows_mobile"
    categoricalColumnNames += "user_knows_devops"
    categoricalColumnNames += "user_knows_data"
    categoricalColumnNames += "user_knows_recsys"
    categoricalColumnNames += "user_is_lead"
    categoricalColumnNames += "user_is_scholar"
    categoricalColumnNames += "user_is_freelancer"
    categoricalColumnNames += "user_is_junior"
    categoricalColumnNames += "user_is_pm"
    categoricalColumnNames += "user_has_blog"
    categoricalColumnNames += "user_binned_company"
    categoricalColumnNames += "user_binned_location"

    listColumnNames += "user_recent_repo_languages"
    listColumnNames += "user_recent_repo_topics"

    textColumnNames += "user_clean_bio"
    textColumnNames += "user_recent_repo_descriptions"

    // Repo Profile

    continuousColumnNames += "repo_size"
    continuousColumnNames += "repo_stargazers_count"
    continuousColumnNames += "repo_forks_count"
    continuousColumnNames += "repo_subscribers_count"
    continuousColumnNames += "repo_open_issues_count"
    continuousColumnNames += "repo_days_between_created_at_today"
    continuousColumnNames += "repo_days_between_updated_at_today"
    continuousColumnNames += "repo_days_between_pushed_at_today"
    continuousColumnNames += "repo_stargazers_subscribers_ratio"
    continuousColumnNames += "repo_stargazers_forks_ratio"

    categoricalColumnNames += "repo_owner_type"
    categoricalColumnNames += "repo_clean_has_issues"
    categoricalColumnNames += "repo_clean_has_projects"
    categoricalColumnNames += "repo_clean_has_downloads"
    categoricalColumnNames += "repo_clean_has_wiki"
    categoricalColumnNames += "repo_clean_has_pages"
    categoricalColumnNames += "repo_is_vinta_starred"
    categoricalColumnNames += "repo_has_homepage"
    categoricalColumnNames += "repo_binned_language"

    listColumnNames += "repo_clean_topics"

    textColumnNames += "repo_text"

    // Split Data

    val weights = if (scala.util.Properties.envOrElse("RUN_ON_SMALL_MACHINE", "false") == "true") Array(0.03, 0.03, 0.94) else Array(0.97, 0.03, 0.0)
    val Array(trainingDF, testDF, _) = featuredDF.randomSplit(weights)
    trainingDF.cache()
    testDF.cache()

    val largeUserIds = testDF.select($"user_id").distinct().map(row => row.getInt(0)).collect().toList
    val sampledUserIds = scala.util.Random.shuffle(largeUserIds).take(500) :+ 652070
    val testUserDF = spark.createDataFrame(sampledUserIds.map(Tuple1(_)))
      .toDF("user_id")
      .cache()

    // Build the Pipeline

    val categoricalTransformers = categoricalColumnNames.flatMap((columnName: String) => {
      val stringIndexer = new StringIndexer()
        .setInputCol(columnName)
        .setOutputCol(s"${columnName}_idx")
        .setHandleInvalid("keep")

      val oneHotEncoder = new OneHotEncoder()
        .setInputCol(s"${columnName}_idx")
        .setOutputCol(s"${columnName}_ohe")
        .setDropLast(false)

      Array(stringIndexer, oneHotEncoder)
    })

    val listTransformers = listColumnNames.flatMap((columnName: String) => {
      val countVectorizerModel = new CountVectorizer()
        .setInputCol(columnName)
        .setOutputCol(s"${columnName}_cv")
        .setMinDF(10)
        .setMinTF(1)

      Array(countVectorizerModel)
    })

    val textTransformers = textColumnNames.flatMap((columnName: String) => {
      val hanLPTokenizer = new HanLPTokenizer()
        .setInputCol(columnName)
        .setOutputCol(s"${columnName}_words")
        .setShouldRemoveStopWords(true)

      val stopWordsRemover = new StopWordsRemover()
        .setInputCol(s"${columnName}_words")
        .setOutputCol(s"${columnName}_filtered_words")
        .setStopWords(StopWordsRemover.loadDefaultStopWords("english"))

      val coreNLPLemmatizer = new CoreNLPLemmatizer()
        .setInputCol(s"${columnName}_filtered_words")
        .setOutputCol(s"${columnName}_lemmatized_words")

      val word2VecModelPath = s"${settings.dataDir}/${settings.today}/word2VecModel.parquet"
      val word2VecModel = Word2VecModel.load(word2VecModelPath)
        .setInputCol(s"${columnName}_lemmatized_words")
        .setOutputCol(s"${columnName}_w2v")

      Array(hanLPTokenizer, stopWordsRemover, coreNLPLemmatizer, word2VecModel)
    })

    val alsModelPath = s"${settings.dataDir}/${settings.today}/alsModel.parquet"
    val alsModel = ALSModel.load(alsModelPath)
      .setUserCol("user_id")
      .setItemCol("repo_id")
      .setPredictionCol("als_score")
      .setColdStartStrategy("drop")

    val finalContinuousColumnNames = (continuousColumnNames :+ "als_score").toArray
    val finalCategoricalColumnNames = categoricalColumnNames.map(columnName => s"${columnName}_ohe").toArray
    val finalListColumnNames = listColumnNames.map(columnName => s"${columnName}_cv").toArray
    val finalTextColumnNames = textColumnNames.map(columnName => s"${columnName}_w2v").toArray
    val vectorAssembler = new VectorAssembler()
      .setInputCols(finalContinuousColumnNames ++ finalCategoricalColumnNames ++ finalListColumnNames ++ finalTextColumnNames)
      .setOutputCol("features")

    val standardScaler = new StandardScaler()
      .setInputCol("features")
      .setOutputCol("standard_features")
      .setWithStd(true)
      .setWithMean(false)

    val sql = """
    SELECT *, CASE WHEN als_score < 0 THEN 0 ELSE als_score END AS weight
    FROM __THIS__
    """.stripMargin
    val weightTransformer = new SQLTransformer().setStatement(sql)

    val intermediateCacher = new IntermediateCacher().setInputCols(Array("standard_features", "weight", "starring"))

    val lr = new LogisticRegression()
      .setMaxIter(100)
      .setRegParam(0.05)
      .setElasticNetParam(0.01)
      .setStandardization(false)
      .setFeaturesCol("standard_features")
      .setLabelCol("starring")
      .setWeightCol("weight")

    val stages = mutable.ArrayBuffer.empty[PipelineStage]
    stages ++= categoricalTransformers
    stages ++= listTransformers
    stages ++= textTransformers
    stages += alsModel
    stages += vectorAssembler
    stages += standardScaler
    stages += weightTransformer
    stages += intermediateCacher
    stages += lr

    val pipeline = new Pipeline().setStages(stages.toArray)

    // Train the Model

    val pipelineModelPath = s"${settings.dataDir}/${settings.today}/rankerPipelineModel.parquet"
    val pipelineModel = loadOrCreateModel[PipelineModel](PipelineModel, pipelineModelPath, () => {
      pipeline.fit(trainingDF)
    })

    // Evaluate the Model: Classification

    val testResultDF = pipelineModel.transform(testDF)

    val binaryClassificationEvaluator = new BinaryClassificationEvaluator()
      .setMetricName("areaUnderROC")
      .setRawPredictionCol("rawPrediction")
      .setLabelCol("starring")

    val classificationMetric = binaryClassificationEvaluator.evaluate(testResultDF)
    println(s"${binaryClassificationEvaluator.getMetricName} = $classificationMetric")

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
      .map((recommender: Recommender) => recommender.recommendForUsers(testUserDF))
      .reduce(_ union _)
      .select($"user_id", $"repo_id")
      .distinct()

    val userCandidateItemDF = userRecommendedItemDF
      .join(userProfileDF, Seq("user_id"))
      .join(repoProfileDF, Seq("repo_id"))

    // Predict the Ranking

    val userRankedItemDF = pipelineModel
      .transform(userCandidateItemDF)
      .cache()

    userRankedItemDF
      .where($"user_id" === 652070)
      .select("user_id", "repo_id", "als_score", "prediction", "probability", "rawPrediction")
      .orderBy(toArrayUDF($"probability").getItem(1).desc)
      .limit(topK)
      .show(false)

    // Evaluate the Model: Ranking

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
    // NDCG@30 = 0.006932765266175483

    spark.stop()
  }
}