package ws.vinta.albedo

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature._
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import ws.vinta.albedo.closures.UserInfoFunctions._
import ws.vinta.albedo.schemas.UserInfo
import ws.vinta.albedo.utils.DatasetUtils.loadUserInfo

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

    // Clean Data

    val cleanUserInfoDS = rawUserInfoDS
      .na.fill("", Array("bio", "blog", "company", "email", "location", "name"))
      .withColumn("clean_company", cleanCompanyUDF($"company"))
      .withColumn("clean_email", cleanEmailUDF($"email"))
      .withColumn("clean_location", cleanLocationUDF($"location"))
      .as[UserInfo]
    cleanUserInfoDS.cache()

    cleanUserInfoDS.show()

    // Feature Engineering

    val continuousColumnNames = Array("public_repos", "public_gists", "followers", "following")

    val categoricalColumnNames = Array("account_type", "clean_company", "clean_email", "clean_location")

    val datetimeColumnNames = Array("created_at", "updated_at")

    val textColumnNames = Array("bio")

    // Feature Transformation

    val companiesDF = cleanUserInfoDS
      .groupBy($"clean_company")
      .agg(count($"user_id").alias("user_count_per_company"))

    val emailsDF = cleanUserInfoDS
      .groupBy($"clean_email")
      .agg(count($"user_id").alias("user_count_per_email"))

    val locationsDF = cleanUserInfoDS
      .groupBy($"clean_location")
      .agg(count($"user_id").alias("user_count_per_location"))

    val cleanUserInfoDFwithCounts = cleanUserInfoDS
      .join(companiesDF, Seq("clean_company"))
      .join(emailsDF, Seq("clean_email"))
      .join(locationsDF, Seq("clean_location"))

    val binnedUserInfoDF = cleanUserInfoDFwithCounts
      .withColumn("binned_company", when($"user_count_per_company" <= 5, "__other").otherwise($"clean_company"))
      .withColumn("binned_email", when($"user_count_per_email" <= 5, "__other").otherwise($"clean_email"))
      .withColumn("binned_location", when($"user_count_per_location" <= 5, "__other").otherwise($"clean_location"))

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

    val textTransformers = textColumnNames.flatMap((columnName: String) => {
      // TODO: 處理中文分詞
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

    // Build the Pipeline

    val userPipeline = new Pipeline()
      .setStages(categoricalTransformers ++ textTransformers)

    val userPipelineModel = userPipeline.fit(cleanUserInfoDS)

    val userInfoDS = userPipelineModel.transform(cleanUserInfoDS).as[UserInfo]

    // Save Results

    userInfoDS.show()

    spark.stop()
  }
}