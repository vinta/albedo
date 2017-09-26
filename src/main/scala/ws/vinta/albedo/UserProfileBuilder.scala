package ws.vinta.albedo

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature._
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import ws.vinta.albedo.closures.UDFs._
import ws.vinta.albedo.utils.DatasetUtils._

import scala.collection.mutable

object UserProfileBuilder {
  def main(args: Array[String]): Unit = {
    implicit val spark: SparkSession = SparkSession
      .builder()
      .appName("UserProfileBuilder")
      .getOrCreate()

    import spark.implicits._

    // Load Data

    val rawUserInfoDS = loadUserInfo()
    rawUserInfoDS.cache()

    val rawRepoInfoDS = loadRepoInfo()
    rawRepoInfoDS.cache()

    val rawRepoStarringDS = loadRepoStarring()
    rawRepoStarringDS.cache()

    // Clean Data

    val nullableColumnNames = Array("bio", "blog", "company", "email", "location", "name")

    val cleanUserInfoDF = rawUserInfoDS
      .withColumn("has_null", when(nullableColumnNames.map(rawUserInfoDS(_).isNull).reduce(_||_), 1.0).otherwise(0.0))
      .na.fill("", nullableColumnNames)
      .withColumn("clean_bio", lower($"bio"))
      .withColumn("clean_company", cleanCompanyUDF($"company"))
      .withColumn("clean_email", cleanEmailUDF($"email"))
      .withColumn("clean_location", cleanLocationUDF($"location"))
    cleanUserInfoDF.cache()

    // Feature Engineering

    var continuousColumnNames = mutable.ArrayBuffer("public_repos", "public_gists", "followers", "following")

    var categoricalColumnNames = mutable.ArrayBuffer("user_id", "account_type", "clean_company", "clean_email", "clean_location")

    var textColumnNames = mutable.ArrayBuffer("clean_bio")

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

    val constructedUserInfoDF = cleanUserInfoDF
      .withColumn("created_at_years_since_today", round(datediff(current_date(), $"created_at") / 365))
      .withColumn("updated_at_days_since_today", datediff(current_date(), $"updated_at"))
      .withColumn("knows_web", containsAnyOfUDF(webThings)($"clean_bio"))
      .withColumn("knows_backend", containsAnyOfUDF(backendThings)($"clean_bio"))
      .withColumn("knows_frontend", containsAnyOfUDF(frontendThings)($"clean_bio"))
      .withColumn("knows_mobile", containsAnyOfUDF(mobileThings)($"clean_bio"))
      .withColumn("knows_devops", containsAnyOfUDF(devopsThings)($"clean_bio"))
      .withColumn("knows_data", containsAnyOfUDF(dataThings)($"clean_bio"))
      .withColumn("knows_recsys", containsAnyOfUDF(recsysThings)($"clean_bio"))
      .withColumn("is_lead", containsAnyOfUDF(leadTitles)($"clean_bio"))
      .withColumn("is_schoolar", containsAnyOfUDF(scholarTitles)($"clean_bio"))
      .withColumn("is_freelancer", containsAnyOfUDF(freelancerTitles)($"clean_bio"))
      .withColumn("is_junior", containsAnyOfUDF(juniorTitles)($"clean_bio"))
      .withColumn("is_pm", containsAnyOfUDF(pmTitles)($"clean_bio"))
      .join(userStarredReposCountDF, Seq("user_id"))
    constructedUserInfoDF.cache()

    continuousColumnNames = continuousColumnNames ++ mutable.ArrayBuffer("created_at_years_since_today", "updated_at_days_since_today", "starred_repos_count")

    categoricalColumnNames = categoricalColumnNames ++ mutable.ArrayBuffer("has_null", "knows_web", "knows_backend", "knows_frontend", "knows_mobile", "knows_devops", "knows_data", "knows_recsys", "is_lead", "is_schoolar", "is_freelancer", "is_junior", "is_pm")

    // Transform Features

    val companiesDF = cleanUserInfoDF
      .groupBy($"clean_company")
      .agg(count("*").alias("count_per_company"))

    val emailsDF = cleanUserInfoDF
      .groupBy($"clean_email")
      .agg(count("*").alias("count_per_email"))

    val locationsDF = cleanUserInfoDF
      .groupBy($"clean_location")
      .agg(count("*").alias("count_per_location"))

    val transformedUserInfoDF = constructedUserInfoDF
      .join(companiesDF, Seq("clean_company"))
      .join(emailsDF, Seq("clean_email"))
      .join(locationsDF, Seq("clean_location"))
      .withColumn("has_website", when($"blog" === "", 0.0).otherwise(1.0))
      .withColumn("binned_company", when($"count_per_company" <= 5, "__other").otherwise($"clean_company"))
      .withColumn("binned_email", when($"count_per_email" <= 2, "__other").otherwise($"clean_email"))
      .withColumn("binned_location", when($"count_per_location" <= 50, "__other").otherwise($"clean_location"))
    transformedUserInfoDF.cache()

    categoricalColumnNames = categoricalColumnNames ++ mutable.ArrayBuffer("has_website", "binned_company", "binned_email", "binned_location")

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

    // Assemble Featuresr

    continuousColumnNames = continuousColumnNames

    categoricalColumnNames = categoricalColumnNames.map((columnName: String) => s"${columnName}_ohe")

    textColumnNames = textColumnNames.map((columnName: String) => s"${columnName}_w2v")

    val vectorAssembler = new VectorAssembler()
      .setInputCols((continuousColumnNames ++ categoricalColumnNames ++ textColumnNames).toArray)
      .setOutputCol("features")

    // Build the Pipeline

    val userPipeline = new Pipeline()
      .setStages((categoricalTransformers ++ textTransformers :+ vectorAssembler).toArray)

    val userPipelineModel = userPipeline.fit(transformedUserInfoDF)

    val userProfileDF = userPipelineModel.transform(transformedUserInfoDF)

    // Save Results

    val pipedUserInfoDFsavePath = s"${settings.dataDir}/${settings.today}/userProfileDF.parquet"
    userProfileDF.write.mode("overwrite").parquet(pipedUserInfoDFsavePath)

    userProfileDF.select("user_id", "features").show()

    spark.stop()
  }
}