package ws.vinta.albedo

import org.apache.spark.SparkConf
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature._
import org.apache.spark.ml.{Pipeline, PipelineModel, Transformer}
import org.apache.spark.ml.recommendation.ALSModel
import org.apache.spark.sql.SparkSession
import ws.vinta.albedo.closures.UDFs._
import ws.vinta.albedo.evaluators.RankingEvaluator
import ws.vinta.albedo.evaluators.RankingEvaluator._
import ws.vinta.albedo.recommenders._
import ws.vinta.albedo.schemas.UserItems
import ws.vinta.albedo.transformers.{HanLPTokenizer, NegativeBalancer}
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
        .setLabelCol("starring")
        .setNegativeValue(0.0)
        .setNegativePositiveRatio(1.0)
      negativeBalancer.transform(rawStarringDS)
    })

    // Feature Engineering

    val continuousColumnNames = mutable.ArrayBuffer.empty[String]
    val categoricalColumnNames = mutable.ArrayBuffer.empty[String]
    val listColumnNames = mutable.ArrayBuffer.empty[String]
    val textColumnNames = mutable.ArrayBuffer.empty[String]

    val featuredDF = balancedStarringDF
      .join(userProfileDF, Seq("user_id"))
      .join(repoProfileDF, Seq("repo_id"))

    featuredDF.show(false)

    categoricalColumnNames += "user_id"
    categoricalColumnNames += "repo_id"

    // User Profile
    continuousColumnNames += "public_repos"
    continuousColumnNames += "public_gists"
    continuousColumnNames += "followers"
    continuousColumnNames += "following"
    continuousColumnNames += "follower_following_ratio"
    continuousColumnNames += "days_between_created_at_today"
    continuousColumnNames += "days_between_updated_at_today"
    continuousColumnNames += "starred_repos_count"
    continuousColumnNames += "avg_daily_starred_repos_count"

    categoricalColumnNames += "account_type"
    categoricalColumnNames += "has_null"
    categoricalColumnNames += "knows_web"
    categoricalColumnNames += "knows_backend"
    categoricalColumnNames += "knows_frontend"
    categoricalColumnNames += "knows_mobile"
    categoricalColumnNames += "knows_devops"
    categoricalColumnNames += "knows_data"
    categoricalColumnNames += "knows_recsys"
    categoricalColumnNames += "is_lead"
    categoricalColumnNames += "is_scholar"
    categoricalColumnNames += "is_freelancer"
    categoricalColumnNames += "is_junior"
    categoricalColumnNames += "is_pm"
    categoricalColumnNames += "has_blog"
    categoricalColumnNames += "binned_company"
    categoricalColumnNames += "binned_location"

    listColumnNames += "top_languages"
    listColumnNames += "top_topics"

    textColumnNames += "clean_bio"
    textColumnNames += "top_descriptions"

    // Repo Profile
    continuousColumnNames += "size"
    continuousColumnNames += "stargazers_count"
    continuousColumnNames += "forks_count"
    continuousColumnNames += "subscribers_count"
    continuousColumnNames += "open_issues_count"
    continuousColumnNames += "days_between_created_at_today"
    continuousColumnNames += "days_between_updated_at_today"
    continuousColumnNames += "days_between_pushed_at_today"
    continuousColumnNames += "stargazers_subscribers_count_ratio"
    continuousColumnNames += "stargazers_forks_count_ratio"

    categoricalColumnNames += "owner_type"
    categoricalColumnNames += "clean_has_issues"
    categoricalColumnNames += "clean_has_projects"
    categoricalColumnNames += "clean_has_downloads"
    categoricalColumnNames += "clean_has_wiki"
    categoricalColumnNames += "clean_has_pages"
    categoricalColumnNames += "is_vinta_starred"
    categoricalColumnNames += "has_homepage"
    categoricalColumnNames += "binned_language"

    listColumnNames += "clean_topics"

    textColumnNames += "text"

    // Split Data

    val trainingTestWeights = if (scala.util.Properties.envOrElse("RUN_ON_SMALL_MACHINE", "false") == "true") Array(0.3, 0.7) else Array(0.8, 0.2)
    val takeN = if (scala.util.Properties.envOrElse("RUN_ON_SMALL_MACHINE", "false") == "true") 100 else 500

    val Array(trainingFeaturedDF, testFeaturedDF) = featuredDF.randomSplit(trainingTestWeights)
    trainingFeaturedDF.cache()
    testFeaturedDF.cache()

    val largeUserIds = testFeaturedDF.select($"user_id").distinct().map(row => row.getInt(0)).collect().toList
    val sampledUserIds = scala.util.Random.shuffle(largeUserIds).take(takeN) :+ 652070
    val testUserDF = spark.createDataFrame(sampledUserIds.map(Tuple1(_))).toDF("user_id")
    testUserDF.cache()

    // Build the Pipeline

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

      val word2VecModel = new Word2Vec()
        .setInputCol(s"${columnName}_words")
        .setOutputCol(s"${columnName}_w2v")
        .setMaxIter(20)
        .setVectorSize(200)
        .setWindowSize(5)
        .setMinCount(10)

      Array(hanLPTokenizer, word2VecModel)
    })

    // TODO: add UDFTransformer()
    // user_repo_follows_repo_owner: 該用戶是否追蹤該 repo 的作者
    // user_repo_starred_language_count: 針對該 repo 所屬的語言，該用戶打星了多少個該語言的 repo
    // user_repo_starred_language_days_until_today: 針對該 repo 的語言，該用戶最近打星任一該語言的 repo 時距離今天幾天

    // TODO: add weightCol
    // .setWeightCol("weight")
    // 讓 positive 的權重高一點
    // 讓新 repo 的權重高一點
    // 有在 top_languages 裡的 repo 權重高一點

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

    val lr = new LogisticRegression()
      .setMaxIter(10)
      .setRegParam(0.0)
      .setElasticNetParam(0.1)
      .setStandardization(false)
      .setFeaturesCol("standard_features")
      .setLabelCol("starring")

    val stages = categoricalTransformers ++ listTransformers ++ textTransformers :+ alsModel :+ vectorAssembler :+ standardScaler :+ lr
    val pipeline = new Pipeline()
      .setStages(stages.toArray)

    // Train the Model

    val pipelineModelPath = s"${settings.dataDir}/${settings.today}/rankerPipelineModel.parquet"
    val pipelineModel = loadOrCreateModel[PipelineModel](PipelineModel, pipelineModelPath, () => {
      pipeline.fit(trainingFeaturedDF)
    })

    // Evaluate

    val testResultDF = pipelineModel.transform(testFeaturedDF)

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
      .select("user_id", "repo_id", "als_score", "prediction", "probability", "rawPrediction")
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
    // NDCG@30 = 0.0014685120623806613

    spark.stop()
  }
}