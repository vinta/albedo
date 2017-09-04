package ws.vinta.albedo

import org.apache.spark.SparkContext
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{BinaryLogisticRegressionSummary, LogisticRegression}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature._
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.SparkSession
import ws.vinta.albedo.preprocessors.{NegativeGenerator, UserInfoTransformer}
import ws.vinta.albedo.utils.DatasetUtils._
import ws.vinta.albedo.utils.StringUtils._

import scala.collection.mutable

object PersonalizedRankerTrainer {
  def main(args: Array[String]): Unit = {
    val activeUser = args(1)
    println(activeUser)

    implicit val spark: SparkSession = SparkSession
      .builder()
      .appName("PersonalizedRankerTrainer")
      .getOrCreate()

    import spark.implicits._

    val sc = spark.sparkContext

    // Load Data

    val rawUserInfoDS = loadUserInfo()
    rawUserInfoDS.cache()

    val rawRepoInfoDS = loadRepoInfo()
    rawRepoInfoDS.cache()

    val rawRepoStarringDS = loadRepoStarring()
    rawRepoStarringDS.cache()

    //val rawUserRelationDS = loadUserRelation()
    //rawUserRelationDS.cache()

    // Impute Missing Values

    val imputedUserInfoDF = rawUserInfoDS.na.fill("", Array("bio", "blog", "company", "email", "location", "name"))

    val imputedRepoInfoDF = rawRepoInfoDS.na.fill("", Array("description", "homepage"))

    // Clean Data

    val cleanUserInfoDF = new UserInfoTransformer().transform(imputedUserInfoDF)
    cleanUserInfoDF.show()

    //val cleanRepoInfoDF = imputedRepoInfoDF.where($"stargazers_count" >= 2)

    //cleanRepoInfoDF.show()

    // Handle Imbalanced Samples

    // TODO: repoStarring join repoInfo join userInfo，過濾掉為 null 的 repoStarring

    val popularReposDF = loadPopularRepos()
    val popularRepos: mutable.LinkedHashSet[Int] = popularReposDF
      .select("repo_id")
      .map(row => row(0).asInstanceOf[Int])
      .collect()
      .to[mutable.LinkedHashSet]
    val bcPopularRepos = sc.broadcast(popularRepos)

    val negativeGenerator = new NegativeGenerator(bcPopularRepos)
    negativeGenerator
      .setUserCol("user_id")
      .setItemCol("repo_id")
      .setLabelCol("starring")
      .setNegativeValue(0.0)
      .setNegativePositiveRatio(1.0)
    val balancedRepoStarringDF = negativeGenerator.transform(rawRepoStarringDS)

    //val fullDF = balanceStarringDF.join(repoInfoDF, Seq("repo_id"))
    //
    //val Array(trainingDF, testDF) = fullDF.randomSplit(Array(0.8, 0.2))

    //// Feature Engineering
    //
    //val countTokensUDF = udf((words: Seq[String]) => words.length)
    //
    //val tempDF = userInfoDF.where("bio != ''")
    //
    //val regexTokenizer = new RegexTokenizer()
    //  .setToLowercase(true)
    //  .setInputCol("bio")
    //  .setOutputCol("bio_words")
    //  .setPattern("\\W").setGaps(true)
    //val tokenizedDF = regexTokenizer.transform(tempDF)
    //
    //// val df1 = tokenizedDF
    ////   .select("bio", "bio_words")
    ////   .withColumn("tokens", countTokensUDF($"bio_words"))
    //
    //val stopWordsRemover = new StopWordsRemover()
    //  .setInputCol("bio_words")
    //  .setOutputCol("bio_filtered_words")
    //val wordsFilteredDF = stopWordsRemover.transform(tokenizedDF)
    //
    //val stringIndexer = new StringIndexer()
    //  .setInputCol("repo_language")
    //  .setOutputCol("repo_language_index")
    //  .setHandleInvalid("keep")
    //val stringIndexerModel = stringIndexer.fit(df1)
    //
    //val indexedDF = stringIndexerModel.transform(df2)
    //indexedDF.show()
    //
    //val word2VecModel = Word2VecModel.load(s"${Settings.dataDir}/20170831/word2VecModel.parquet")
    //
    //// Build the Pipeline
    //
    //// Train the Model
    //
    //val fullDF = balanceStarringDF.join(repoInfoDF, balanceStarringDF.col("repo_id") === rawRepoInfoDS.col("id"))
    //
    //val Array(trainingDF, testDF) = fullDF.randomSplit(Array(0.8, 0.2))
    //
    //val vectorAssembler = new VectorAssembler()
    //  .setInputCols(Array("size", "stargazers_count", "forks_count", "subscribers_count"))
    //  .setOutputCol("features")
    //
    //val vectorTrainingDF = vectorAssembler.transform(trainingDF)
    //val vectorTestDF = vectorAssembler.transform(testDF)
    //
    //val lr = new LogisticRegression()
    //  .setMaxIter(30)
    //  .setRegParam(0.0)
    //  .setElasticNetParam(0.0)
    //  .setFeaturesCol("features")
    //  .setLabelCol("starring")
    //
    //val lrModel = lr.fit(vectorTrainingDF)
    //
    //val pipeline = new Pipeline()
    //  .setStages(stringIndexerModels ++ oneHotEncoders :+ vectorAssembler)
    ////  .setStages(stringIndexerModels ++ oneHotEncoders :+ vectorAssembler :+ lr)
    //
    //val pipelineModel = pipeline.fit(trainingDF)
    //
    //val resultTrainingDF = pipelineModel.transform(trainingDF)
    //val resultTestDF = pipelineModel.transform(testDF)
    //
    //// Show the Model Summary
    //
    //println(s"Coefficients: ${lrModel.coefficients}")
    //println(s"Intercept: ${lrModel.intercept}")
    //
    //val modelSummary = lrModel.summary.asInstanceOf[BinaryLogisticRegressionSummary]
    //println(s"Area Under ROC: ${modelSummary.areaUnderROC}")
    //
    //// Evaluate the Model
    //
    //val resultTestDF = lrModel.transform(vectorTestDF)
    //
    //val evaluator = new BinaryClassificationEvaluator()
    //  .setMetricName("areaUnderROC")
    //  .setRawPredictionCol("rawPrediction")
    //  .setLabelCol("starring")
    //val metric = evaluator.evaluate(resultTestDF)
    //println(s"${evaluator.getMetricName}: $metric")
    //
    //// Predict
    //
    //resultTestDF.where("user_id = 652070").select("user_id", "repo_id", "starring", "prediction", "probability").show(false)
    //
    //import org.apache.spark.ml.linalg.{Vector, Vectors}
    //
    //val to_array = udf((v: Vector) => v.toDense.values)
    //
    //resultTestDF
    //  .where("user_id = 652070")
    //  .orderBy(to_array($"probability").getItem(1).desc)
    //  .select("user_id", "repo_id", "starring", "prediction", "probability")
    //  .show(false)
    //
    //// Rank

    spark.stop()
  }
}