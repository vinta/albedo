package ws.vinta.albedo

import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions._
import ws.vinta.albedo.closures.UDFs._
import ws.vinta.albedo.utils.DatasetUtils._

import scala.collection.mutable

object UserProfileBuilder {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf()
    if (scala.util.Properties.envOrElse("RUN_WITH_INTELLIJ", "false") == "true") {
      conf.setMaster("local[*]")
      conf.set("spark.driver.memory", "12g")
      //conf.setMaster("local-cluster[1, 3, 12288]")
      //conf.setJars(List("target/albedo-1.0.0-SNAPSHOT-uber.jar"))
      //conf.setMaster("spark://localhost:7077")
      //conf.set("spark.driver.memory", "2g")
      //conf.set("spark.executor.cores", "3")
      //conf.set("spark.executor.memory", "12g")
      //conf.setJars(List("target/albedo-1.0.0-SNAPSHOT-uber.jar"))
    }

    implicit val spark: SparkSession = SparkSession
      .builder()
      .appName("UserProfileBuilder")
      .config(conf)
      .getOrCreate()

    import spark.implicits._

    // Load Data

    val rawUserInfoDS = loadRawUserInfoDS().cache()

    val rawRepoInfoDS = loadRawRepoInfoDS().cache()

    val rawStarringDS = loadRawStarringDS().cache()

    // Feature Engineering

    val booleanColumnNames = mutable.ArrayBuffer.empty[String]
    val continuousColumnNames = mutable.ArrayBuffer.empty[String]
    val categoricalColumnNames = mutable.ArrayBuffer.empty[String]
    val listColumnNames = mutable.ArrayBuffer.empty[String]
    val textColumnNames = mutable.ArrayBuffer.empty[String]

    continuousColumnNames += "user_public_repos_count"
    continuousColumnNames += "user_public_gists_count"
    continuousColumnNames += "user_followers_count"
    continuousColumnNames += "user_following_count"

    categoricalColumnNames += "user_account_type"

    // Impute Data

    val nullableColumnNames = Array("user_name", "user_company", "user_blog", "user_location", "user_bio")

    val imputedUserInfoDF = rawUserInfoDS
      .withColumn("user_has_null", when(nullableColumnNames.map(rawUserInfoDS(_).isNull).reduce(_ || _), true).otherwise(false))
      .na.fill("", nullableColumnNames)

    booleanColumnNames += "user_has_null"

    // Clean Data

    val cleanUserInfoDF = imputedUserInfoDF
      .withColumn("user_clean_company", cleanCompanyUDF($"user_company"))
      .withColumn("user_clean_location", cleanLocationUDF($"user_location"))
      .withColumn("user_clean_bio", lower($"user_bio"))
      .cache()

    textColumnNames += "user_clean_bio"

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

    val userStarredReposCountDF = rawStarringDS
      .groupBy($"user_id")
      .agg(count("*").alias("user_starred_repos_count"))
      .cache()

    val starringRepoInfoDF = rawStarringDS
      .select($"user_id", $"repo_id", $"starred_at")
      .join(rawRepoInfoDS, Seq("repo_id"))
      .cache()

    val userTopLanguagesDF = starringRepoInfoDF
      .withColumn("rank", rank.over(Window.partitionBy($"user_id").orderBy($"starred_at".desc)))
      .where($"rank" <= 50)
      .groupBy($"user_id")
      .agg(collect_list(lower($"repo_language")).alias("user_recent_repo_languages"))
      .select($"user_id", $"user_recent_repo_languages")

    val userTopTopicsDF = starringRepoInfoDF
      .where($"repo_topics" =!= "")
      .withColumn("rank", rank.over(Window.partitionBy($"user_id").orderBy($"starred_at".desc)))
      .where($"rank" <= 50)
      .groupBy($"user_id")
      .agg(concat_ws(",", collect_list(lower($"repo_topics"))).alias("temp_user_recent_repo_topics"))
      .select($"user_id", split($"temp_user_recent_repo_topics", ",").alias("user_recent_repo_topics"))

    val userTopDescriptionDF = starringRepoInfoDF
      .where($"repo_description" =!= "")
      .withColumn("rank", rank.over(Window.partitionBy($"user_id").orderBy($"starred_at".desc)))
      .where($"rank" <= 50)
      .groupBy($"user_id")
      .agg(concat_ws(" ", collect_list(lower($"repo_description"))).alias("user_recent_repo_descriptions"))
      .select($"user_id", $"user_recent_repo_descriptions")

    val constructedUserInfoDF = cleanUserInfoDF
      .withColumn("user_knows_web", when(webThings.map($"user_clean_bio".like(_)).reduce(_ or _), true).otherwise(false))
      .withColumn("user_knows_backend", when(backendThings.map($"user_clean_bio".like(_)).reduce(_ or _), true).otherwise(false))
      .withColumn("user_knows_frontend", when(frontendThings.map($"user_clean_bio".like(_)).reduce(_ or _), true).otherwise(false))
      .withColumn("user_knows_mobile", when(mobileThings.map($"user_clean_bio".like(_)).reduce(_ or _), true).otherwise(false))
      .withColumn("user_knows_devops", when(devopsThings.map($"user_clean_bio".like(_)).reduce(_ or _), true).otherwise(false))
      .withColumn("user_knows_data", when(dataThings.map($"user_clean_bio".like(_)).reduce(_ or _), true).otherwise(false))
      .withColumn("user_knows_recsys", when(recsysThings.map($"user_clean_bio".like(_)).reduce(_ or _), true).otherwise(false))
      .withColumn("user_is_lead", when(leadTitles.map($"user_clean_bio".like(_)).reduce(_ or _), true).otherwise(false))
      .withColumn("user_is_scholar", when(scholarTitles.map($"user_clean_bio".like(_)).reduce(_ or _), true).otherwise(false))
      .withColumn("user_is_freelancer", when(freelancerTitles.map($"user_clean_bio".like(_)).reduce(_ or _), true).otherwise(false))
      .withColumn("user_is_junior", when(juniorTitles.map($"user_clean_bio".like(_)).reduce(_ or _), true).otherwise(false))
      .withColumn("user_is_pm", when(pmTitles.map($"user_clean_bio".like(_)).reduce(_ or _), true).otherwise(false))
      .withColumn("user_followers_following_ratio", round($"user_followers_count" / ($"user_following_count" + lit(1.0)), 3))
      .withColumn("user_days_between_created_at_today", datediff(current_date(), $"user_created_at"))
      .withColumn("user_days_between_updated_at_today", datediff(current_date(), $"user_updated_at"))
      .join(userStarredReposCountDF, Seq("user_id"))
      .cache()
      .withColumn("user_avg_daily_starred_repos_count", round($"user_starred_repos_count" / ($"user_days_between_created_at_today" + lit(1.0)), 3))
      .join(userTopDescriptionDF, Seq("user_id"))
      .join(userTopTopicsDF, Seq("user_id"))
      .join(userTopLanguagesDF, Seq("user_id"))
      .cache()

    booleanColumnNames += "user_knows_web"
    booleanColumnNames += "user_knows_backend"
    booleanColumnNames += "user_knows_frontend"
    booleanColumnNames += "user_knows_mobile"
    booleanColumnNames += "user_knows_devops"
    booleanColumnNames += "user_knows_data"
    booleanColumnNames += "user_knows_recsys"
    booleanColumnNames += "user_is_lead"
    booleanColumnNames += "user_is_scholar"
    booleanColumnNames += "user_is_freelancer"
    booleanColumnNames += "user_is_junior"
    booleanColumnNames += "user_is_pm"

    continuousColumnNames += "user_followers_following_ratio"
    continuousColumnNames += "user_days_between_created_at_today"
    continuousColumnNames += "user_days_between_updated_at_today"
    continuousColumnNames += "user_starred_repos_count"
    continuousColumnNames += "user_avg_daily_starred_repos_count"

    listColumnNames += "user_recent_repo_languages"
    listColumnNames += "user_recent_repo_topics"

    textColumnNames += "user_recent_repo_descriptions"

    // Convert Features

    val companyCountDF = cleanUserInfoDF
      .groupBy($"user_clean_company")
      .agg(count("*").alias("count_per_user_company"))
      .select($"user_clean_company", $"count_per_user_company")
      .cache()

    val locationCountDF = cleanUserInfoDF
      .groupBy($"user_clean_location")
      .agg(count("*").alias("count_per_user_location"))
      .select($"user_clean_location", $"count_per_user_location")
      .cache()

    val transformedUserInfoDF = constructedUserInfoDF
      .join(companyCountDF, Seq("user_clean_company"))
      .join(locationCountDF, Seq("user_clean_location"))
      .withColumn("user_has_blog", when($"user_blog" === "", false).otherwise(true))
      .withColumn("user_binned_company", when($"count_per_user_company" <= 5, "__other").otherwise($"user_clean_company"))
      .withColumn("user_binned_location", when($"count_per_user_location" <= 50, "__other").otherwise($"user_clean_location"))
      .cache()

    booleanColumnNames += "user_has_blog"

    categoricalColumnNames += "user_binned_company"
    categoricalColumnNames += "user_binned_location"

    // Save Results

    // Boolean column names: user_has_null, user_knows_web, user_knows_backend, user_knows_frontend, user_knows_mobile, user_knows_devops, user_knows_data, user_knows_recsys, user_is_lead, user_is_scholar, user_is_freelancer, user_is_junior, user_is_pm, user_has_blog
    // Continuous column names: user_public_repos_count, user_public_gists_count, user_followers_count, user_following_count, user_followers_following_ratio, user_days_between_created_at_today, user_days_between_updated_at_today, user_starred_repos_count, user_avg_daily_starred_repos_count
    // Categorical column names: user_account_type, user_binned_company, user_binned_location
    // List column names: user_recent_repo_languages, user_recent_repo_topics
    // Text column names: user_clean_bio, user_recent_repo_descriptions
    println("Boolean column names: " + booleanColumnNames.mkString(", "))
    println("Continuous column names: " + continuousColumnNames.mkString(", "))
    println("Categorical column names: " + categoricalColumnNames.mkString(", "))
    println("List column names: " + listColumnNames.mkString(", "))
    println("Text column names: " + textColumnNames.mkString(", "))

    val featureNames = mutable.ArrayBuffer("user_id", "user_login")
    featureNames ++= booleanColumnNames
    featureNames ++= continuousColumnNames
    featureNames ++= categoricalColumnNames
    featureNames ++= listColumnNames
    featureNames ++= textColumnNames

    val path = s"${settings.dataDir}/${settings.today}/userProfileDF.parquet"
    val userProfileDF = loadOrCreateDataFrame(path, () => {
      transformedUserInfoDF.select(featureNames.map(col): _*)
    })

    userProfileDF.show(false)

    spark.stop()
  }
}