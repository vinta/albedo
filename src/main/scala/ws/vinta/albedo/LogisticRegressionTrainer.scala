package ws.vinta.albedo

import org.apache.spark.ml.classification.{BinaryLogisticRegressionSummary, LogisticRegression}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.SparkSession
import ws.vinta.albedo.preprocessors.{NegativeGenerator, popularReposBuilder}
import ws.vinta.albedo.utils.DataSourceUtils.{loadRepoInfo, loadRepoStarring, loadUserInfo, loadUserRelation}

import scala.collection.mutable

object LogisticRegressionTrainer {
  val appName = "LogisticRegressionTrainer"

  def main(args: Array[String]): Unit = {
    val activeUser = args(1)
    println(activeUser)

    implicit val spark = SparkSession
      .builder()
      .appName(appName)
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

    // Handle Imbalanced Samples

    val popularReposDF = popularReposBuilder.transform(rawRepoInfoDS)
    val popularRepos: mutable.LinkedHashSet[Int] = popularReposDF
      .select("id")
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

    // Feature Engineering

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

    // Predict

    // Rank

    spark.stop()
  }
}