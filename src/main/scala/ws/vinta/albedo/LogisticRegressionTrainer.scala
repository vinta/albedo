package ws.vinta.albedo

import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.sql.SparkSession
import ws.vinta.albedo.preprocessors.{NegativeGenerator, popularReposBuilder}
import ws.vinta.albedo.schemas.PopularRepo
import ws.vinta.albedo.utils.DataSourceUtils.{loadRepoInfo, loadRepoStarring}

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

    // load data

    val repoInfoDS = loadRepoInfo()
    repoInfoDS.cache()

    val repoStarringDS = loadRepoStarring()
    repoStarringDS.cache()
    println(repoStarringDS.count())

    // handle imbalanced samples

    val popularReposDS = popularReposBuilder
      .transform(repoInfoDS)
      .as[PopularRepo]
    val popularRepos: mutable.LinkedHashSet[Int] = popularReposDS
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
      .setNegativeValue(0)
      .setNegativePositiveRatio(1.0)
    val balancedDF = negativeGenerator.transform(repoStarringDS)
    println(balancedDF.count())

    // train the model

    val fullDF = balancedDF.join(repoInfoDS, balancedDF.col("repo_id") === repoInfoDS.col("id"))

    val Array(training, test) = fullDF.randomSplit(Array(0.8, 0.2))

    import org.apache.spark.ml.feature.VectorAssembler

    val vectorAssembler = new VectorAssembler()
      .setInputCols(Array("stargazers_count", "forks_count", "subscribers_count"))
      .setOutputCol("features")

    val outputDF = vectorAssembler.transform(training)

    val lr = new LogisticRegression()
      .setMaxIter(10)
      .setRegParam(0.3)
      .setElasticNetParam(0.8)
      .setFeaturesCol("features")
      .setLabelCol("starring")

    val lrModel = lr.fit(outputDF)
    println(s"Coefficients: ${lrModel.coefficients}")
    println(s"Intercept: ${lrModel.intercept}")

    spark.stop()
  }
}