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
      .withColumn("user_has_null", when(nullableColumnNames.map(rawUserInfoDS(_).isNull).reduce(_ || _), 1.0).otherwise(0.0))
      .na.fill("", nullableColumnNames)

    categoricalColumnNames += "user_has_null"

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

    val starringRepoInfoDF = rawStarringDS
      .join(rawRepoInfoDS, Seq("repo_id"))
      .cache()

    val userTopLanguagesDF = starringRepoInfoDF
      .withColumn("rank", rank.over(Window.partitionBy($"user_id").orderBy($"starred_at".desc)))
      .where($"rank" <= 50)
      .groupBy($"user_id")
      .agg(collect_list($"repo_language").alias("user_recent_repo_languages"))
      .select($"user_id", $"user_recent_repo_languages")

    val userTopTopicsDF = starringRepoInfoDF
      .where($"repo_topics" =!= "")
      .withColumn("rank", rank.over(Window.partitionBy($"user_id").orderBy($"starred_at".desc)))
      .where($"rank" <= 50)
      .groupBy($"user_id")
      .agg(concat_ws(",", collect_list($"repo_topics")).alias("temp_user_recent_repo_topics"))
      .select($"user_id", split($"temp_user_recent_repo_topics", ",").alias("user_recent_repo_topics"))

    val userTopDescriptionDF = starringRepoInfoDF
      .where($"repo_description" =!= "")
      .withColumn("rank", rank.over(Window.partitionBy($"user_id").orderBy($"starred_at".desc)))
      .where($"rank" <= 50)
      .groupBy($"user_id")
      .agg(concat_ws(" ", collect_list($"repo_description")).alias("user_recent_repo_descriptions"))
      .select($"user_id", $"user_recent_repo_descriptions")

    val constructedUserInfoDF = cleanUserInfoDF
      .withColumn("user_followers_following_ratio", round($"user_followers_count" / ($"user_following_count" + lit(1)), 3))
      .withColumn("user_days_between_created_at_today", datediff(current_date(), $"user_created_at"))
      .withColumn("user_days_between_updated_at_today", datediff(current_date(), $"user_updated_at"))
      .withColumn("user_knows_web", when(webThings.map($"user_clean_bio".like(_)).reduce(_ or _), 1.0).otherwise(0.0))
      .withColumn("user_knows_backend", when(backendThings.map($"user_clean_bio".like(_)).reduce(_ or _), 1.0).otherwise(0.0))
      .withColumn("user_knows_frontend", when(frontendThings.map($"user_clean_bio".like(_)).reduce(_ or _), 1.0).otherwise(0.0))
      .withColumn("user_knows_mobile", when(mobileThings.map($"user_clean_bio".like(_)).reduce(_ or _), 1.0).otherwise(0.0))
      .withColumn("user_knows_devops", when(devopsThings.map($"user_clean_bio".like(_)).reduce(_ or _), 1.0).otherwise(0.0))
      .withColumn("user_knows_data", when(dataThings.map($"user_clean_bio".like(_)).reduce(_ or _), 1.0).otherwise(0.0))
      .withColumn("user_knows_recsys", when(recsysThings.map($"user_clean_bio".like(_)).reduce(_ or _), 1.0).otherwise(0.0))
      .withColumn("user_is_lead", when(leadTitles.map($"user_clean_bio".like(_)).reduce(_ or _), 1.0).otherwise(0.0))
      .withColumn("user_is_scholar", when(scholarTitles.map($"user_clean_bio".like(_)).reduce(_ or _), 1.0).otherwise(0.0))
      .withColumn("user_is_freelancer", when(freelancerTitles.map($"user_clean_bio".like(_)).reduce(_ or _), 1.0).otherwise(0.0))
      .withColumn("user_is_junior", when(juniorTitles.map($"user_clean_bio".like(_)).reduce(_ or _), 1.0).otherwise(0.0))
      .withColumn("user_is_pm", when(pmTitles.map($"user_clean_bio".like(_)).reduce(_ or _), 1.0).otherwise(0.0))
      .join(userStarredReposCountDF, Seq("user_id"))
      .withColumn("user_avg_daily_starred_repos_count", round($"user_starred_repos_count" / ($"user_days_between_created_at_today" + lit(1)), 3))
      .join(userTopDescriptionDF, Seq("user_id"))
      .join(userTopTopicsDF, Seq("user_id"))
      .join(userTopLanguagesDF, Seq("user_id"))
      .cache()

    continuousColumnNames += "user_followers_following_ratio"
    continuousColumnNames += "user_days_between_created_at_today"
    continuousColumnNames += "user_days_between_updated_at_today"
    continuousColumnNames += "user_starred_repos_count"
    continuousColumnNames += "user_avg_daily_starred_repos_count"

    categoricalColumnNames += "user_knows_web"
    categoricalColumnNames += "user_knows_backend"
    categoricalColumnNames += "user_knows_frontend"
    categoricalColumnNames += "user_knows_mobile"
    categoricalColumnNames += "user_knows_devops"
    categoricalColumnNames += "user_knows_data"
    categoricalColumnNames += "user_knows_recsys"
    categoricalColumnNames += "user_is_lead"
    categoricalColumnNames += "user_is_scholar"
    categoricalColumnNames += "user_is_freelancer"
    categoricalColumnNames += "user_is_junior"
    categoricalColumnNames += "user_is_pm"

    listColumnNames += "user_recent_repo_languages"
    listColumnNames += "user_recent_repo_topics"

    textColumnNames += "user_recent_repo_descriptions"

    // Transform Features

    val companyCountDF = cleanUserInfoDF
      .groupBy($"user_clean_company")
      .agg(count("*").alias("count_per_user_company"))

    val locationCountDF = cleanUserInfoDF
      .groupBy($"user_clean_location")
      .agg(count("*").alias("count_per_user_location"))

    val transformedUserInfoDF = constructedUserInfoDF
      .join(companyCountDF, Seq("user_clean_company"))
      .join(locationCountDF, Seq("user_clean_location"))
      .withColumn("user_has_blog", when($"user_blog" === "", 0.0).otherwise(1.0))
      .withColumn("user_binned_company", when($"count_per_user_company" <= 5, "__other").otherwise($"user_clean_company"))
      .withColumn("user_binned_location", when($"count_per_user_location" <= 50, "__other").otherwise($"user_clean_location"))

    categoricalColumnNames += "user_has_blog"
    categoricalColumnNames += "user_binned_company"
    categoricalColumnNames += "user_binned_location"

    // Save Results

    // Continuous column names: user_public_repos_count, user_public_gists_count, user_followers_count, user_following_count, user_followers_following_ratio, user_days_between_created_at_today, user_days_between_updated_at_today, user_starred_repos_count, user_avg_daily_starred_repos_count
    // Categorical column names: user_account_type, user_has_null, user_knows_web, user_knows_backend, user_knows_frontend, user_knows_mobile, user_knows_devops, user_knows_data, user_knows_recsys, user_is_lead, user_is_scholar, user_is_freelancer, user_is_junior, user_is_pm, user_has_blog, user_binned_company, user_binned_location
    // List column names: user_recent_repo_languages, user_recent_repo_topics
    // Text column names: user_clean_bio, user_recent_repo_descriptions
    println("Continuous column names: " + continuousColumnNames.mkString(", "))
    println("Categorical column names: " + categoricalColumnNames.mkString(", "))
    println("List column names: " + listColumnNames.mkString(", "))
    println("Text column names: " + textColumnNames.mkString(", "))

    val featureNames = mutable.ArrayBuffer("user_id", "user_login") ++ continuousColumnNames ++ categoricalColumnNames ++ listColumnNames ++ textColumnNames
    val path = s"${settings.dataDir}/${settings.today}/userProfileDF.parquet"
    val userProfileDF = loadOrCreateDataFrame(path, () => {
      transformedUserInfoDF.select(featureNames.map(col): _*)
    })

    userProfileDF.show(false)

    spark.stop()
  }
}