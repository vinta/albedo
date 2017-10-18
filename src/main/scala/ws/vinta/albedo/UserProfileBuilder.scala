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
    conf.set("spark.driver.memory", "4g")
    conf.set("spark.executor.memory", "12g")
    conf.set("spark.executor.cores", "4")

    implicit val spark: SparkSession = SparkSession
      .builder()
      .appName("UserProfileBuilder")
      .config(conf)
      .getOrCreate()

    import spark.implicits._

    // Load Data

    val rawUserInfoDS = loadRawUserInfoDS()
    rawUserInfoDS.cache()

    val rawRepoInfoDS = loadRawRepoInfoDS()
    rawRepoInfoDS.cache()

    val rawStarringDS = loadRawStarringDS()
    rawStarringDS.cache()

    // Feature Engineering

    val continuousColumnNames = mutable.ArrayBuffer.empty[String]
    val categoricalColumnNames = mutable.ArrayBuffer.empty[String]
    val listColumnNames = mutable.ArrayBuffer.empty[String]
    val textColumnNames = mutable.ArrayBuffer.empty[String]

    continuousColumnNames += "public_repos"
    continuousColumnNames += "public_gists"
    continuousColumnNames += "followers"
    continuousColumnNames += "following"

    categoricalColumnNames += "account_type"

    // Impute Data

    val nullableColumnNames = Array("bio", "blog", "company", "location", "name")

    val imputedUserInfoDF = rawUserInfoDS
      .withColumn("has_null", when(nullableColumnNames.map(rawUserInfoDS(_).isNull).reduce(_ || _), 1.0).otherwise(0.0))
      .na.fill("", nullableColumnNames)

    categoricalColumnNames += "has_null"

    // Clean Data

    val cleanUserInfoDF = imputedUserInfoDF
      .withColumn("clean_bio", lower($"bio"))
      .withColumn("clean_company", cleanCompanyUDF($"company"))
      .withColumn("clean_location", cleanLocationUDF($"location"))
    cleanUserInfoDF.cache()

    textColumnNames += "clean_bio"

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
      .agg(count("*").alias("starred_repos_count"))

    val starringRepoInfoDF = rawStarringDS.join(rawRepoInfoDS, Seq("repo_id"))
    starringRepoInfoDF.cache()

    val userTopLanguagesDF = starringRepoInfoDF
      .withColumn("rank", rank.over(Window.partitionBy($"user_id").orderBy($"starred_at".desc)))
      .where($"rank" <= 50)
      .groupBy($"user_id")
      .agg(collect_list($"language").alias("top_languages"))
      .select($"user_id", $"top_languages")

    val userTopTopicsDF = starringRepoInfoDF
      .where($"topics" =!= "")
      .withColumn("rank", rank.over(Window.partitionBy($"user_id").orderBy($"starred_at".desc)))
      .where($"rank" <= 50)
      .groupBy($"user_id")
      .agg(concat_ws(",", collect_list($"topics")).alias("topics_concat"))
      .select($"user_id", split($"topics_concat", ",").alias("top_topics"))

    val userTopDescriptionDF = starringRepoInfoDF
      .where($"description" =!= "")
      .withColumn("rank", rank.over(Window.partitionBy($"user_id").orderBy($"starred_at".desc)))
      .where($"rank" <= 50)
      .groupBy($"user_id")
      .agg(concat_ws(" ", collect_list($"description")).alias("top_descriptions"))
      .select($"user_id", $"top_descriptions")

    val constructedUserInfoDF = cleanUserInfoDF
      .withColumn("follower_following_ratio", round($"followers" / ($"following" + lit(1)), 3))
      .withColumn("days_between_created_at_today", datediff(current_date(), $"created_at"))
      .withColumn("days_between_updated_at_today", datediff(current_date(), $"updated_at"))
      .withColumn("knows_web", when(webThings.map($"clean_bio".like(_)).reduce(_ or _), 1.0).otherwise(0.0))
      .withColumn("knows_backend", when(backendThings.map($"clean_bio".like(_)).reduce(_ or _), 1.0).otherwise(0.0))
      .withColumn("knows_frontend", when(frontendThings.map($"clean_bio".like(_)).reduce(_ or _), 1.0).otherwise(0.0))
      .withColumn("knows_mobile", when(mobileThings.map($"clean_bio".like(_)).reduce(_ or _), 1.0).otherwise(0.0))
      .withColumn("knows_devops", when(devopsThings.map($"clean_bio".like(_)).reduce(_ or _), 1.0).otherwise(0.0))
      .withColumn("knows_data", when(dataThings.map($"clean_bio".like(_)).reduce(_ or _), 1.0).otherwise(0.0))
      .withColumn("knows_recsys", when(recsysThings.map($"clean_bio".like(_)).reduce(_ or _), 1.0).otherwise(0.0))
      .withColumn("is_lead", when(leadTitles.map($"clean_bio".like(_)).reduce(_ or _), 1.0).otherwise(0.0))
      .withColumn("is_scholar", when(scholarTitles.map($"clean_bio".like(_)).reduce(_ or _), 1.0).otherwise(0.0))
      .withColumn("is_freelancer", when(freelancerTitles.map($"clean_bio".like(_)).reduce(_ or _), 1.0).otherwise(0.0))
      .withColumn("is_junior", when(juniorTitles.map($"clean_bio".like(_)).reduce(_ or _), 1.0).otherwise(0.0))
      .withColumn("is_pm", when(pmTitles.map($"clean_bio".like(_)).reduce(_ or _), 1.0).otherwise(0.0))
      .join(userStarredReposCountDF, Seq("user_id"))
      .withColumn("avg_daily_starred_repos_count", round($"starred_repos_count" / ($"days_between_created_at_today" + lit(1)), 3))
      .join(userTopDescriptionDF, Seq("user_id"))
      .join(userTopTopicsDF, Seq("user_id"))
      .join(userTopLanguagesDF, Seq("user_id"))
    constructedUserInfoDF.cache()

    continuousColumnNames += "follower_following_ratio"
    continuousColumnNames += "days_between_created_at_today"
    continuousColumnNames += "days_between_updated_at_today"
    continuousColumnNames += "starred_repos_count"
    continuousColumnNames += "avg_daily_starred_repos_count"

    categoricalColumnNames += "knows_web"
    categoricalColumnNames += "knows_backend"
    categoricalColumnNames += "knows_frontend"
    categoricalColumnNames += "knows_mobile"
    categoricalColumnNames += "knows_devops"
    categoricalColumnNames += "knows_data"
    categoricalColumnNames += "knows_recsys"
    categoricalColumnNames += "is_lead"
    categoricalColumnNames += "is_scholar"
    categoricalColumnNames += "is_freelancer"
    categoricalColumnNames += "is_junior"
    categoricalColumnNames += "is_pm"

    listColumnNames += "top_languages"
    listColumnNames += "top_topics"

    textColumnNames += "top_descriptions"

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
      .withColumn("has_blog", when($"blog" === "", 0.0).otherwise(1.0))
      .withColumn("binned_company", when($"count_per_company" <= 5, "__other").otherwise($"clean_company"))
      .withColumn("binned_location", when($"count_per_location" <= 50, "__other").otherwise($"clean_location"))

    categoricalColumnNames += "has_blog"
    categoricalColumnNames += "binned_company"
    categoricalColumnNames += "binned_location"

    // Save Results

    // Continuous column names: public_repos, public_gists, followers, following, follower_following_ratio, days_between_created_at_today, days_between_updated_at_today, starred_repos_count, avg_daily_starred_repos_count
    // Categorical column names: account_type, has_null, knows_web, knows_backend, knows_frontend, knows_mobile, knows_devops, knows_data, knows_recsys, is_lead, is_scholar, is_freelancer, is_junior, is_pm, has_blog, binned_company, binned_location
    // List column names: top_languages, top_topics
    // Text column names: clean_bio, top_descriptions
    println("Continuous column names: " + continuousColumnNames.mkString(", "))
    println("Categorical column names: " + categoricalColumnNames.mkString(", "))
    println("List column names: " + listColumnNames.mkString(", "))
    println("Text column names: " + textColumnNames.mkString(", "))

    val featureNames = mutable.ArrayBuffer("user_id", "login") ++ continuousColumnNames ++ categoricalColumnNames ++ listColumnNames ++ textColumnNames
    val path = s"${settings.dataDir}/${settings.today}/userProfileDF.parquet"
    val userProfileDF = loadOrCreateDataFrame(path, () => {
      transformedUserInfoDF.select(featureNames.map(col): _*)
    })

    userProfileDF.show(false)

    spark.stop()
  }
}