package ws.vinta.albedo

import org.apache.spark.SparkConf
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, SparkSession}
import ws.vinta.albedo.transformers.HanLPTokenizer
import ws.vinta.albedo.utils.DatasetUtils._

import scala.collection.mutable

object RepoProfileBuilder {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf()
    conf.set("spark.driver.memory", "4g")
    conf.set("spark.executor.memory", "12g")
    conf.set("spark.executor.cores", "4")

    implicit val spark: SparkSession = SparkSession
      .builder()
      .appName("RepoProfileBuilder")
      .config(conf)
      .getOrCreate()

    val sc = spark.sparkContext
    sc.setCheckpointDir("./spark-data/checkpoint")

    import spark.implicits._

    // Load Data

    val rawRepoInfoDS = loadRawRepoInfoDS()
    rawRepoInfoDS.cache()

    val rawUserInfoDS = loadRawUserInfoDS()
    rawUserInfoDS.cache()

    val rawStarringDS = loadRawStarringDS()
    rawStarringDS.cache()

    // Clean Data

    val nullableColumnNames = Array("description", "homepage")

    val lowerableColumnNames = Array("description", "language", "topics")

    val booleanColumnNames = Array("fork", "has_issues", "has_projects", "has_downloads", "has_wiki", "has_pages")

    val filledRepoInfoDF = rawRepoInfoDS
      .where($"stargazers_count".between(2, 100000))
      .where($"forks_count" <= 90000)
      .where($"fork" === false)
      .na.fill("", nullableColumnNames)

    val cleanRepoInfoDF = (lowerableColumnNames ++ booleanColumnNames).foldLeft[DataFrame](filledRepoInfoDF)((accDF, columnName) => {
      columnName match {
        case _ if lowerableColumnNames.contains(columnName) =>
          accDF.withColumn(s"clean_$columnName", lower(col(columnName)))
        case _ if booleanColumnNames.contains(columnName) =>
          accDF.withColumn(s"clean_$columnName", col(columnName).cast("int"))
      }
    })
    cleanRepoInfoDF.cache()

    // Feature Engineering

    var continuousColumnNames = mutable.ArrayBuffer("stargazers_count", "forks_count", "subscribers_count", "open_issues_count")

    var categoricalColumnNames = mutable.ArrayBuffer("owner_type") ++ booleanColumnNames.map(columnName => s"clean_$columnName")

    val textColumnNames = mutable.ArrayBuffer("clean_description", "clean_topics")

    // Construct Features

    val unmaintainedWords = Array("%unmaintained%", "%no longer maintained%", "%no longer actively maintained%", "%not maintained%", "%not actively maintained%", "%deprecated%", "%moved to%")
    val assignmentWords = Array("%assignment%")

    val vintaStarredRepos = rawStarringDS
      .where($"user_id" === 652070)
      .select($"repo_id".as[Int])
      .collect()
      .to[List]

    val constructedRepoInfoDF = cleanRepoInfoDF
      .withColumn("is_unmaintained", when(unmaintainedWords.map($"clean_description".like(_)).reduce(_ or _), 1.0).otherwise(0.0))
      .withColumn("is_assignment", when(assignmentWords.map($"clean_description".like(_)).reduce(_ or _), 1.0).otherwise(0.0))
      .where($"is_unmaintained" === 0 and $"is_assignment" === 0)
      .drop($"is_unmaintained")
      .drop($"is_assignment")
      .withColumn("created_at_years_since_today", round(datediff(current_date(), $"created_at") / 365))
      .withColumn("updated_at_days_since_today", datediff(current_date(), $"updated_at"))
      .withColumn("pushed_at_days_since_today", datediff(current_date(), $"pushed_at"))
      .withColumn("watch_star_count_ratio", round($"subscribers_count" / ($"stargazers_count" + lit(1)), 3))
      .withColumn("fork_star_count_ratio", round($"forks_count" / ($"stargazers_count" + lit(1)), 3))
      .withColumn("is_vinta_starred", when($"repo_id".isin(vintaStarredRepos: _*), 1.0).otherwise(0.0))

    continuousColumnNames = continuousColumnNames ++ mutable.ArrayBuffer("created_at_years_since_today", "updated_at_days_since_today", "pushed_at_days_since_today")

    // Transform Features

    val languagesDF = cleanRepoInfoDF
      .groupBy($"clean_language")
      .agg(count("*").alias("count_per_language"))

    val transformedRepoInfoDF = constructedRepoInfoDF
      .join(languagesDF, Seq("clean_language"))
      .withColumn("has_homepage", when($"homepage" === "", 0.0).otherwise(1.0))
      .withColumn("binned_language", when($"count_per_language" <= 30, "__other").otherwise($"clean_language"))
    transformedRepoInfoDF.cache()

    categoricalColumnNames = categoricalColumnNames ++ mutable.ArrayBuffer("has_homepage", "binned_language")

    // Categorical Features

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

    // Text Features

    val textTransformers = textColumnNames.flatMap((columnName: String) => {
      val hanLPTokenizer = new HanLPTokenizer()
        .setInputCol(columnName)
        .setOutputCol(s"${columnName}_words")

      val word2VecModel = Word2VecModel.load(s"${settings.dataDir}/${settings.today}/corpusPipelineModel.parquet")
        .setInputCol(s"${columnName}_words")
        .setOutputCol(s"${columnName}_w2v")

      Array(hanLPTokenizer, word2VecModel)
    })

    // Assemble Features

    val finalContinuousColumnNames = continuousColumnNames

    val finalCategoricalColumnNames = categoricalColumnNames.map((columnName: String) => s"${columnName}_ohe")

    val finalTextColumnNames = textColumnNames.map((columnName: String) => s"${columnName}_w2v")

    val vectorAssembler = new VectorAssembler()
      .setInputCols((finalContinuousColumnNames ++ finalCategoricalColumnNames ++ finalTextColumnNames).toArray)
      .setOutputCol("features")

    // Build the Pipeline

    val repoPipeline = new Pipeline()
      .setStages((categoricalTransformers ++ textTransformers :+ vectorAssembler).toArray)

    val repoPipelineModel = repoPipeline.fit(transformedRepoInfoDF)

    // Save Results

    val path = s"${settings.dataDir}/${settings.today}/repoProfileDF.parquet"
    val repoProfileDF = loadOrCreateDataFrame(path, () => {
      repoPipelineModel.transform(transformedRepoInfoDF)
    })

    // features length: 528
    repoProfileDF.select("repo_id", "full_name", "features").show(false)

    spark.stop()
  }
}