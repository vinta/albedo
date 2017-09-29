package ws.vinta.albedo

import org.apache.spark.SparkConf
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature._
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{AnalysisException, SparkSession}
import ws.vinta.albedo.closures.UDFs._
import ws.vinta.albedo.transformers.HanLPTokenizer
import ws.vinta.albedo.utils.DatasetUtils._

import scala.collection.mutable

object UserProfileBuilder {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf()
    conf.set("spark.driver.memory", "4g")
    conf.set("spark.executor.memory", "12g")
    conf.set("spark.executor.cores", "4")

    implicit val spark: SparkSession = SparkSession
      .builder()
      .appName("UserProfileBuilder")
      .config(conf)
      .getOrCreate()

    val sc = spark.sparkContext
    sc.setCheckpointDir("./spark-data/checkpoint")

    import spark.implicits._

    // Load Data

    val rawUserInfoDS = loadRawUserInfoDS()
    rawUserInfoDS.cache()

    val rawRepoInfoDS = loadRawRepoInfoDS()
    rawRepoInfoDS.cache()

    val rawRepoStarringDS = loadRawRepoStarringDS()
    rawRepoStarringDS.cache()

    // Clean Data

    val nullableColumnNames = Array("bio", "blog", "company", "location", "name")

    val cleanUserInfoDF = rawUserInfoDS
      .withColumn("has_null", when(nullableColumnNames.map(rawUserInfoDS(_).isNull).reduce(_||_), 1).otherwise(0))
      .na.fill("", nullableColumnNames)
      .withColumn("clean_bio", lower($"bio"))
      .withColumn("clean_company", cleanCompanyUDF($"company"))
      .withColumn("clean_location", cleanLocationUDF($"location"))
    cleanUserInfoDF.cache()

    // Feature Engineering

    var continuousColumnNames = mutable.ArrayBuffer("public_repos", "public_gists", "followers", "following")

    var categoricalColumnNames = mutable.ArrayBuffer("account_type")

    var listColumnNames = mutable.ArrayBuffer.empty[String]

    val textColumnNames = mutable.ArrayBuffer("clean_bio")

    // Construct Features

    val webThings = Array("web", "fullstack", "full stack")
    val backendThings = Array("backend", "back end", "back-end")
    val frontendThings = Array("frontend", "front end", "front-end")
    val mobileThings = Array("mobile", "ios", "android")
    val devopsThings = Array("devops", "sre", "admin", "infrastructure")
    val dataThings = Array("machine learning", "deep learning", "data scien", "data analy")
    val recsysThings = Array("data mining", "recommend", "information retrieval")

    val leadTitles = Array("team lead", "architect", "creator", "director", "cto", "vp of engineering")
    val scholarTitles = Array("researcher", "scientist", "phd", "professor")
    val freelancerTitles = Array("freelance")
    val juniorTitles = Array("junior", "beginner", "newbie")
    val pmTitles = Array("product manager")

    val userStarredReposCountDF = rawRepoStarringDS
      .groupBy($"user_id")
      .agg(count("*").alias("starred_repos_count"))

    val repoInfoStarringDF = rawRepoStarringDS.join(rawRepoInfoDS, Seq("repo_id"))
    repoInfoStarringDF.cache()

    val userLanguagesDF = repoInfoStarringDF
      .withColumn("rank", rank.over(Window.partitionBy($"user_id").orderBy($"starred_at".desc)))
      .where($"rank" <= 50)
      .groupBy($"user_id")
      .agg(collect_list($"language").alias("languages_preferences"))

    val userTopicsDF = repoInfoStarringDF
      .where($"topics" =!= "")
      .withColumn("rank", rank.over(Window.partitionBy($"user_id").orderBy($"starred_at".desc)))
      .where($"rank" <= 50)
      .groupBy($"user_id")
      .agg(concat_ws(",", collect_list($"topics")).alias("topics_concat"))
      .select($"user_id", split($"topics_concat", ",").alias("topics_preferences"))

    val constructedUserInfoDF = cleanUserInfoDF
      .withColumn("created_at_years_since_today", round(datediff(current_date(), $"created_at") / 365))
      .withColumn("updated_at_days_since_today", datediff(current_date(), $"updated_at"))
      .withColumn("knows_web", when(webThings.map($"clean_bio".like(_)).reduce(_ or _), 1).otherwise(0))
      .withColumn("knows_backend", when(backendThings.map($"clean_bio".like(_)).reduce(_ or _), 1).otherwise(0))
      .withColumn("knows_frontend", when(frontendThings.map($"clean_bio".like(_)).reduce(_ or _), 1).otherwise(0))
      .withColumn("knows_mobile", when(mobileThings.map($"clean_bio".like(_)).reduce(_ or _), 1).otherwise(0))
      .withColumn("knows_devops", when(devopsThings.map($"clean_bio".like(_)).reduce(_ or _), 1).otherwise(0))
      .withColumn("knows_data", when(dataThings.map($"clean_bio".like(_)).reduce(_ or _), 1).otherwise(0))
      .withColumn("knows_recsys", when(recsysThings.map($"clean_bio".like(_)).reduce(_ or _), 1).otherwise(0))
      .withColumn("is_lead", when(leadTitles.map($"clean_bio".like(_)).reduce(_ or _), 1).otherwise(0))
      .withColumn("is_schoolar", when(scholarTitles.map($"clean_bio".like(_)).reduce(_ or _), 1).otherwise(0))
      .withColumn("is_freelancer", when(freelancerTitles.map($"clean_bio".like(_)).reduce(_ or _), 1).otherwise(0))
      .withColumn("is_junior", when(juniorTitles.map($"clean_bio".like(_)).reduce(_ or _), 1).otherwise(0))
      .withColumn("is_pm", when(pmTitles.map($"clean_bio".like(_)).reduce(_ or _), 1).otherwise(0))
      .join(userStarredReposCountDF, Seq("user_id"))
      .join(userLanguagesDF, Seq("user_id"))
      .join(userTopicsDF, Seq("user_id"))
    constructedUserInfoDF.cache()

    continuousColumnNames = continuousColumnNames ++ mutable.ArrayBuffer("created_at_years_since_today", "updated_at_days_since_today", "starred_repos_count")

    categoricalColumnNames = categoricalColumnNames ++ mutable.ArrayBuffer("has_null", "knows_web", "knows_backend", "knows_frontend", "knows_mobile", "knows_devops", "knows_data", "knows_recsys", "is_lead", "is_schoolar", "is_freelancer", "is_junior", "is_pm")

    listColumnNames = listColumnNames ++ mutable.ArrayBuffer("languages_preferences", "topics_preferences")

    // Transform Features

    val companyCountDF = cleanUserInfoDF
      .groupBy($"clean_company")
      .agg(count("*").alias("count_per_company"))

    val locationCountDF = cleanUserInfoDF
      .groupBy($"clean_location")
      .agg(count("*").alias("count_per_location"))

    val transformedUserInfoDF = constructedUserInfoDF
      .join(companyCountDF, Seq("clean_company"))
      .join(locationCountDF, Seq("clean_location"))
      .withColumn("has_blog", when($"blog" === "", 0).otherwise(1))
      .withColumn("binned_company", when($"count_per_company" <= 5, "__other").otherwise($"clean_company"))
      .withColumn("binned_location", when($"count_per_location" <= 50, "__other").otherwise($"clean_location"))
    transformedUserInfoDF.cache()

    categoricalColumnNames = categoricalColumnNames ++ mutable.ArrayBuffer("has_blog", "binned_company", "binned_location")

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

    // List Features

    val listTransformers = listColumnNames.flatMap((columnName: String) => {
      val word2VecModel = new Word2Vec()
        .setInputCol(columnName)
        .setOutputCol(s"${columnName}_w2v")
        .setMaxIter(10)
        .setVectorSize(100)
        .setWindowSize(5)
        .setMinCount(1)

      Array(word2VecModel)
    })

    // Text Features

    val textTransformers = textColumnNames.flatMap((columnName: String) => {
      val hanLPTokenizer = new HanLPTokenizer()
        .setInputCol(columnName)
        .setOutputCol(s"${columnName}_words")

      val word2VecModel = Word2VecModel.load(s"${settings.dataDir}/20170903/word2VecModelCorpus.parquet")
        .setInputCol(s"${columnName}_words")
        .setOutputCol(s"${columnName}_w2v")

      Array(hanLPTokenizer, word2VecModel)
    })

    // Assemble Features

    val finalContinuousColumnNames = continuousColumnNames

    val finalCategoricalColumnNames = categoricalColumnNames.map((columnName: String) => s"${columnName}_ohe")

    val finalListColumnNames = listColumnNames.map((columnName: String) => s"${columnName}_w2v")

    val finalTextColumnNames = textColumnNames.map((columnName: String) => s"${columnName}_w2v")

    val vectorAssembler = new VectorAssembler()
      .setInputCols((finalContinuousColumnNames ++ finalCategoricalColumnNames ++ finalListColumnNames ++ finalTextColumnNames).toArray)
      .setOutputCol("features")

    // Build the Pipeline

    val userPipeline = new Pipeline()
      .setStages((categoricalTransformers ++ listTransformers ++ textTransformers :+ vectorAssembler).toArray)

    val userPipelineModel = userPipeline.fit(transformedUserInfoDF)

    // Save Results

    val savePath = s"${settings.dataDir}/${settings.today}/userProfileDF.parquet"
    val userProfileDF = try {
      spark.read.parquet(savePath)
    } catch {
      case e: AnalysisException => {
        if (e.getMessage().contains("Path does not exist")) {
          val df = userPipelineModel.transform(transformedUserInfoDF)
          df.write.mode("overwrite").parquet(savePath)
          df
        } else {
          throw e
        }
      }
    }

    // features length: 923
    userProfileDF.select("user_id", "login", "features").show(false)

    spark.stop()
  }
}