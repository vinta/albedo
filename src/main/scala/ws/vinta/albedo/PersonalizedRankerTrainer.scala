package ws.vinta.albedo

import org.apache.spark.ml.classification.{BinaryLogisticRegressionSummary, LogisticRegression}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{VectorAssembler, Word2VecModel}
import org.apache.spark.sql.SparkSession
import ws.vinta.albedo.preprocessors.{NegativeGenerator, popularReposBuilder}
import ws.vinta.albedo.utils.DataSourceUtils.{loadRepoInfo, loadRepoStarring, loadUserInfo, loadUserRelation}
import ws.vinta.albedo.utils.Settings

import scala.collection.mutable

object PersonalizedRankerTrainer {
  def main(args: Array[String]): Unit = {
    val activeUser = args(1)
    println(activeUser)

    implicit val spark = SparkSession
      .builder()
      .appName("LogisticRegressionTrainer")
      .getOrCreate()

    implicit val sc = spark.sparkContext
    sc.setLogLevel("WARN")

    import spark.implicits._

    // Load Data

    val rawUserInfoDS = loadUserInfo()
    rawUserInfoDS.cache()

    val rawUserRelationDS = loadUserRelation()
    rawUserRelationDS.cache()

    val rawRepoInfoDS = loadRepoInfo()
    rawRepoInfoDS.cache()

    val rawRepoStarringDS = loadRepoStarring()
    rawRepoStarringDS.cache()

    // Impute Missing Values

    // Clean Data

    // TODO: 過濾掉 star 數 <= 1 的 repo

    // Handle Imbalanced Samples

    // TODO: repoStarring join repoInfo join userInfo，過濾掉為 null 的 repoStarring

    val popularReposDF = popularReposBuilder.transform(rawRepoInfoDS)
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
    val balanceStarringDF = negativeGenerator.transform(rawRepoStarringDS)

    // Impute Missing Values

    val repoInfoDF = rawRepoInfoDS.na.fill("", Array("description", "homepage"))
    repoInfoDF.cache()

    // 移除 repoStarring 中 user id 和 item id 不在 userInfo 和 repoInfo 的紀錄

    // Feature Engineering

    import org.apache.spark.ml.feature.StringIndexer
    import org.apache.spark.ml.feature.OneHotEncoder

    //val stringIndexer = new StringIndexer()
    //  .setInputCol("repo_language")
    //  .setOutputCol("repo_language_index")
    //  .setHandleInvalid("keep")
    //val stringIndexerModel = stringIndexer.fit(df1)
    //
    //val indexedDF = stringIndexerModel.transform(df2)
    //indexedDF.show()

    val word2VecModel = Word2VecModel.load(s"${Settings.dataDir}/20170831/word2VecModel.parquet")

    // Train a Model

    val fullDF = balanceStarringDF.join(repoInfoDF, balanceStarringDF.col("repo_id") === rawRepoInfoDS.col("id"))

    val Array(trainingDF, testDF) = fullDF.randomSplit(Array(0.8, 0.2))

    val vectorAssembler = new VectorAssembler()
      .setInputCols(Array("size", "stargazers_count", "forks_count", "subscribers_count"))
      .setOutputCol("features")

    val vectorTrainingDF = vectorAssembler.transform(trainingDF)
    val vectorTestDF = vectorAssembler.transform(testDF)

    val lr = new LogisticRegression()
      .setMaxIter(30)
      .setRegParam(0.0)
      .setElasticNetParam(0.0)
      .setFeaturesCol("features")
      .setLabelCol("starring")

    val lrModel = lr.fit(vectorTrainingDF)

    // Show the Model Summary

    println(s"Coefficients: ${lrModel.coefficients}")
    println(s"Intercept: ${lrModel.intercept}")

    val modelSummary = lrModel.summary.asInstanceOf[BinaryLogisticRegressionSummary]
    println(s"Area Under ROC: ${modelSummary.areaUnderROC}")

    // Evaluate the Model

    //val resultTestDF = lrModel.transform(vectorTestDF)
    //
    //val evaluator = new BinaryClassificationEvaluator()
    //  .setMetricName("areaUnderROC")
    //  .setRawPredictionCol("rawPrediction")
    //  .setLabelCol("starring")
    //val metric = evaluator.evaluate(resultTestDF)
    //println(s"${evaluator.getMetricName}: $metric")

    // Predict

    // Rank

    spark.stop()
  }
}