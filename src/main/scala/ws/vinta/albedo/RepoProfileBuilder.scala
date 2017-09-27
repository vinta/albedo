package ws.vinta.albedo

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, SparkSession}
import ws.vinta.albedo.utils.DatasetUtils._

import scala.collection.mutable

object RepoProfileBuilder {
  def main(args: Array[String]): Unit = {
    implicit val spark: SparkSession = SparkSession
      .builder()
      .appName("RepoProfileBuilder")
      .getOrCreate()

    import spark.implicits._

    val sc = spark.sparkContext
    sc.setCheckpointDir("./spark-data/checkpoint")

    // Load Data

    val rawRepoInfoDS = loadRepoInfo()
    rawRepoInfoDS.cache()

    val rawRepoStarringDS = loadRepoStarring()
    rawRepoStarringDS.cache()

    val rawUserInfoDS = loadUserInfo()
    rawUserInfoDS.cache()

    // Clean Data

    val nullableColumnNames = Array("description", "homepage")

    val lowerableColumnNames = Array("description", "language", "topics")

    val booleanColumnNames = Array("fork", "has_issues", "has_projects", "has_downloads", "has_wiki", "has_pages")

    val filledRepoInfoDF = rawRepoInfoDS
      .where($"stargazers_count".between(2, 100000) and $"forks_count" <= 90000)
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

    val unmaintainedWords = Array("%unmaintained%", "%no longer maintained%", "%no longer actively maintained%", "%not maintained%", "%not actively maintained%", "%deprecated%")
    val assignmentWords = Array("%assignment%")

    val constructedRepoInfoDF = cleanRepoInfoDF
      .withColumn("created_at_years_since_today", round(datediff(current_date(), $"created_at") / 365))
      .withColumn("updated_at_days_since_today", datediff(current_date(), $"updated_at"))
      .withColumn("pushed_at_days_since_today", datediff(current_date(), $"pushed_at"))
      .withColumn("is_unmaintained", when(unmaintainedWords.map($"clean_description".like(_)).reduce(_ or _), 1).otherwise(0))
      .withColumn("is_assignment", when(assignmentWords.map($"clean_description".like(_)).reduce(_ or _), 1).otherwise(0))
      .where($"is_unmaintained" === 0 and $"is_assignment" === 0)
      .drop($"is_unmaintained")
      .drop($"is_assignment")

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

    // Continuous Features

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
      val regexTokenizer = new RegexTokenizer()
        .setToLowercase(true)
        .setInputCol(columnName)
        .setOutputCol(s"${columnName}_words")
        .setPattern("""[\w\-_]+""").setGaps(false)

      val stopWords = StopWordsRemover.loadDefaultStopWords("english")
      val stopWordsRemover = new StopWordsRemover()
        .setStopWords(stopWords)
        .setInputCol(s"${columnName}_words")
        .setOutputCol(s"${columnName}_filtered_words")

      val word2VecModel = Word2VecModel.load(s"${settings.dataDir}/20170903/word2VecModel.parquet")
        .setInputCol(s"${columnName}_filtered_words")
        .setOutputCol(s"${columnName}_w2v")

      Array(regexTokenizer, stopWordsRemover, word2VecModel)
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

    val repoProfileDF = repoPipelineModel.transform(transformedRepoInfoDF)

    // Save Results

    val savePath = s"${settings.dataDir}/${settings.today}/repoProfileDF.parquet"
    repoProfileDF.write.mode("overwrite").parquet(savePath)

    repoProfileDF.select("repo_id", "full_name", "features").show(false)

    spark.stop()
  }
}