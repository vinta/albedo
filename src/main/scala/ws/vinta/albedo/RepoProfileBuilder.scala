package ws.vinta.albedo

import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import ws.vinta.albedo.utils.DatasetUtils._

import scala.collection.mutable

object RepoProfileBuilder {
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
      .appName("RepoProfileBuilder")
      .config(conf)
      .getOrCreate()

    import spark.implicits._

    // Load Data

    val rawRepoInfoDS = loadRawRepoInfoDS().cache()

    val rawStarringDS = loadRawStarringDS().cache()

    // Feature Engineering

    val booleanColumnNames = mutable.ArrayBuffer.empty[String]
    val continuousColumnNames = mutable.ArrayBuffer.empty[String]
    val categoricalColumnNames = mutable.ArrayBuffer.empty[String]
    val listColumnNames = mutable.ArrayBuffer.empty[String]
    val textColumnNames = mutable.ArrayBuffer.empty[String]

    booleanColumnNames += "repo_has_issues"
    booleanColumnNames += "repo_has_projects"
    booleanColumnNames += "repo_has_downloads"
    booleanColumnNames += "repo_has_wiki"
    booleanColumnNames += "repo_has_pages"

    continuousColumnNames += "repo_size"
    continuousColumnNames += "repo_stargazers_count"
    continuousColumnNames += "repo_forks_count"
    continuousColumnNames += "repo_subscribers_count"
    continuousColumnNames += "repo_open_issues_count"

    categoricalColumnNames += "repo_owner_type"

    // Impute Data

    val nullableColumnNames = Array("repo_description", "repo_homepage")

    val imputedRepoInfoDF = rawRepoInfoDS
      .withColumn("repo_has_null", when(nullableColumnNames.map(rawRepoInfoDS(_).isNull).reduce(_ || _), true).otherwise(false))
      .na.fill("", nullableColumnNames)
      .cache()

    booleanColumnNames += "repo_has_null"

    // Clean Data

    val reducedRepoInfo = imputedRepoInfoDF
      .where($"repo_is_fork" === false)
      .where($"repo_forks_count" <= 90000)
      .where($"repo_stargazers_count".between(30, 100000))
      .cache()

    val unmaintainedWords = Array("%unmaintained%", "%no longer maintained%", "%no longer actively maintained%", "%not maintained%", "%not actively maintained%", "%deprecated%", "%moved to%")
    val assignmentWords = Array("%assignment%", "%作業%", "%作业%")
    val demoWords = Array("test", "%demo project%")
    val blogWords = Array("my blog")

    val cleanRepoInfoDF = reducedRepoInfo
      .withColumn("repo_clean_description", lower($"repo_description"))
      .cache()
      .withColumn("repo_is_unmaintained", when(unmaintainedWords.map($"repo_clean_description".like(_)).reduce(_ or _), true).otherwise(false))
      .withColumn("repo_is_assignment", when(assignmentWords.map($"repo_clean_description".like(_)).reduce(_ or _), true).otherwise(false))
      .withColumn("repo_is_demo", when(demoWords.map($"repo_clean_description".like(_)).reduce(_ or _) and $"repo_stargazers_count" <= 40, true).otherwise(false))
      .withColumn("repo_is_blog", when(blogWords.map($"repo_clean_description".like(_)).reduce(_ or _) and $"repo_stargazers_count" <= 40, true).otherwise(false))
      .where($"repo_is_unmaintained" === false)
      .where($"repo_is_assignment" === false)
      .where($"repo_is_demo" === false)
      .where($"repo_is_blog" === false)
      .withColumn("repo_clean_language", lower($"repo_language"))
      .withColumn("repo_clean_topics", lower($"repo_topics"))
      .cache()

    // Construct Features

    val vintaStarredRepos = rawStarringDS
      .where($"user_id" === 652070)
      .select($"repo_id".as[Int])
      .collect()
      .to[List]

    val constructedRepoInfoDF = cleanRepoInfoDF
      .withColumn("repo_has_activities_in_60days", datediff(current_date(), $"repo_pushed_at") <= 60)
      .withColumn("repo_has_homepage", when($"repo_homepage" === "", false).otherwise(true))
      .withColumn("repo_is_vinta_starred", when($"repo_id".isin(vintaStarredRepos: _*), true).otherwise(false))
      .withColumn("repo_days_between_created_at_today", datediff(current_date(), $"repo_created_at"))
      .withColumn("repo_days_between_updated_at_today", datediff(current_date(), $"repo_updated_at"))
      .withColumn("repo_days_between_pushed_at_today", datediff(current_date(), $"repo_pushed_at"))
      .withColumn("repo_subscribers_stargazers_ratio", round($"repo_subscribers_count" / ($"repo_stargazers_count" + lit(1.0)), 3))
      .withColumn("repo_forks_stargazers_ratio", round($"repo_forks_count" / ($"repo_stargazers_count" + lit(1.0)), 3))
      .withColumn("repo_open_issues_stargazers_ratio", round($"repo_open_issues_count" / ($"repo_stargazers_count" + lit(1.0)), 3))
      .withColumn("repo_text", lower(concat_ws(" ", $"repo_owner_username", $"repo_name", $"repo_language", $"repo_description")))

    booleanColumnNames += "repo_has_activities_in_60days"
    booleanColumnNames += "repo_has_homepage"
    booleanColumnNames += "repo_is_vinta_starred"

    continuousColumnNames += "repo_days_between_created_at_today"
    continuousColumnNames += "repo_days_between_updated_at_today"
    continuousColumnNames += "repo_days_between_pushed_at_today"
    continuousColumnNames += "repo_subscribers_stargazers_ratio"
    continuousColumnNames += "repo_forks_stargazers_ratio"
    continuousColumnNames += "repo_open_issues_stargazers_ratio"

    textColumnNames += "repo_text"

    val languagesDF = cleanRepoInfoDF
      .groupBy($"repo_clean_language")
      .agg(count("*").alias("count_per_repo_language"))

    val transformedRepoInfoDF = constructedRepoInfoDF
      .join(languagesDF, Seq("repo_clean_language"))
      .withColumn("repo_binned_language", when($"count_per_repo_language" <= 30, "__other").otherwise($"repo_clean_language"))
      .withColumn("repo_clean_topics", split($"repo_topics", ","))
      .cache()

    categoricalColumnNames += "repo_binned_language"

    listColumnNames += "repo_clean_topics"

    // Save Results

    // Boolean column names: repo_has_issues, repo_has_projects, repo_has_downloads, repo_has_wiki, repo_has_pages, repo_has_null, repo_has_activities_in_60days, repo_has_homepage, repo_is_vinta_starred
    // Continuous column names: repo_size, repo_stargazers_count, repo_forks_count, repo_subscribers_count, repo_open_issues_count, repo_days_between_created_at_today, repo_days_between_updated_at_today, repo_days_between_pushed_at_today, repo_subscribers_stargazers_ratio, repo_forks_stargazers_ratio, repo_open_issues_stargazers_ratio
    // Categorical column names: repo_owner_type, repo_binned_language
    // List column names: repo_clean_topics
    // Text column names: repo_text
    println("Boolean column names: " + booleanColumnNames.mkString(", "))
    println("Continuous column names: " + continuousColumnNames.mkString(", "))
    println("Categorical column names: " + categoricalColumnNames.mkString(", "))
    println("List column names: " + listColumnNames.mkString(", "))
    println("Text column names: " + textColumnNames.mkString(", "))

    val featureNames = mutable.ArrayBuffer("repo_id", "repo_full_name", "repo_owner_id") ++ continuousColumnNames ++ categoricalColumnNames ++ listColumnNames ++ textColumnNames
    val path = s"${settings.dataDir}/${settings.today}/repoProfileDF.parquet"
    val repoProfileDF = loadOrCreateDataFrame(path, () => {
      transformedRepoInfoDF.select(featureNames.map(col): _*)
    })

    repoProfileDF.show(false)

    spark.stop()
  }
}