package ws.vinta.albedo

import org.apache.spark.SparkConf
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, SparkSession}
import ws.vinta.albedo.utils.DatasetUtils._

import scala.collection.mutable

object RepoProfileBuilder {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf()
    if (scala.util.Properties.envOrElse("RUN_ON_SMALL_MACHINE", "false") == "true") {
      conf.set("spark.driver.memory", "4g")
      conf.set("spark.executor.memory", "12g")
      conf.set("spark.executor.cores", "4")
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

    val continuousColumnNames = mutable.ArrayBuffer.empty[String]
    val categoricalColumnNames = mutable.ArrayBuffer.empty[String]
    val listColumnNames = mutable.ArrayBuffer.empty[String]
    val textColumnNames = mutable.ArrayBuffer.empty[String]

    continuousColumnNames += "repo_size"
    continuousColumnNames += "repo_stargazers_count"
    continuousColumnNames += "repo_forks_count"
    continuousColumnNames += "repo_subscribers_count"
    continuousColumnNames += "repo_open_issues_count"

    categoricalColumnNames += "repo_owner_type"

    // Impute Data

    val nullableColumnNames = Array("repo_description", "repo_homepage")

    val imputedRepoInfoDF = rawRepoInfoDS
      .withColumn("repo_has_null", when(nullableColumnNames.map(rawRepoInfoDS(_).isNull).reduce(_ || _), 1.0).otherwise(0.0))
      .na.fill("", nullableColumnNames)

    categoricalColumnNames += "repo_has_null"

    // Clean Data

    val unmaintainedWords = Array("%unmaintained%", "%no longer maintained%", "%no longer actively maintained%", "%not maintained%", "%not actively maintained%", "%deprecated%", "%moved to%")
    val assignmentWords = Array("%assignment%")

    val reducedRepoInfo = imputedRepoInfoDF
      .where($"repo_is_fork" === false)
      .where($"repo_forks_count" <= 90000)
      .where($"repo_stargazers_count".between(10, 100000))
      .withColumn("repo_is_unmaintained", when(unmaintainedWords.map($"repo_clean_description".like(_)).reduce(_ or _), 1.0).otherwise(0.0))
      .withColumn("repo_is_assignment", when(assignmentWords.map($"repo_clean_description".like(_)).reduce(_ or _), 1.0).otherwise(0.0))
      .where($"repo_is_unmaintained" === 0 and $"repo_is_assignment" === 0)

    val lowerableColumnNames = Array("repo_description", "repo_language", "repo_topics")
    val booleanColumnNames = Array("repo_has_issues", "repo_has_projects", "repo_has_downloads", "repo_has_wiki", "repo_has_pages")

    val cleanRepoInfoDF = (lowerableColumnNames ++ booleanColumnNames).foldLeft[DataFrame](reducedRepoInfo)((accDF, columnName) => {
      columnName match {
        case _ if lowerableColumnNames.contains(columnName) =>
          accDF.withColumn(columnName.replaceFirst("repo_", "repo_clean_"), lower(col(columnName)))
        case _ if booleanColumnNames.contains(columnName) =>
          accDF.withColumn(columnName.replaceFirst("repo_", "repo_clean_"), col(columnName).cast("double"))
      }
    })
    .cache()

    categoricalColumnNames += "repo_clean_has_issues"
    categoricalColumnNames += "repo_clean_has_projects"
    categoricalColumnNames += "repo_clean_has_downloads"
    categoricalColumnNames += "repo_clean_has_wiki"
    categoricalColumnNames += "repo_clean_has_pages"

    // Construct Features

    val vintaStarredRepos = rawStarringDS
      .where($"user_id" === 652070)
      .select($"repo_id".as[Int])
      .collect()
      .to[List]

    val constructedRepoInfoDF = cleanRepoInfoDF
      .withColumn("repo_days_between_created_at_today", datediff(current_date(), $"repo_created_at"))
      .withColumn("repo_days_between_updated_at_today", datediff(current_date(), $"repo_updated_at"))
      .withColumn("repo_days_between_pushed_at_today", datediff(current_date(), $"repo_pushed_at"))
      .withColumn("repo_stargazers_subscribers_ratio", round($"repo_stargazers_count" / ($"repo_subscribers_count" + lit(1)), 3))
      .withColumn("repo_stargazers_forks_ratio", round($"repo_stargazers_count" / ($"repo_forks_count" + lit(1)), 3))
      .withColumn("repo_is_vinta_starred", when($"repo_id".isin(vintaStarredRepos: _*), 1.0).otherwise(0.0))
      .withColumn("repo_text", concat_ws(" ", $"repo_owner_username", $"repo_name", $"repo_language", $"repo_description"))

    continuousColumnNames += "repo_days_between_created_at_today"
    continuousColumnNames += "repo_days_between_updated_at_today"
    continuousColumnNames += "repo_days_between_pushed_at_today"
    continuousColumnNames += "repo_stargazers_subscribers_ratio"
    continuousColumnNames += "repo_stargazers_forks_ratio"

    categoricalColumnNames += "repo_is_vinta_starred"

    textColumnNames += "repo_text"

    // Transform Features

    val languagesDF = cleanRepoInfoDF
      .groupBy($"repo_clean_language")
      .agg(count("*").alias("count_per_repo_language"))

    val transformedRepoInfoDF = constructedRepoInfoDF
      .join(languagesDF, Seq("repo_clean_language"))
      .withColumn("repo_has_homepage", when($"repo_homepage" === "", 0.0).otherwise(1.0))
      .withColumn("repo_binned_language", when($"count_per_repo_language" <= 30, "__other").otherwise($"repo_clean_language"))
      .withColumn("repo_clean_topics", split($"repo_topics", ","))
      .cache()

    categoricalColumnNames += "repo_has_homepage"
    categoricalColumnNames += "repo_binned_language"

    listColumnNames += "repo_clean_topics"

    // Save Results

    // Continuous column names: repo_size, repo_stargazers_count, repo_forks_count, repo_subscribers_count, repo_open_issues_count, repo_days_between_created_at_today, repo_days_between_updated_at_today, repo_days_between_pushed_at_today, repo_stargazers_subscribers_ratio, repo_stargazers_forks_ratio
    // Categorical column names: repo_owner_type, repo_clean_has_issues, repo_clean_has_projects, repo_clean_has_downloads, repo_clean_has_wiki, repo_clean_has_pages, repo_is_vinta_starred, repo_has_homepage, repo_binned_language
    // List column names: repo_clean_topics
    // Text column names: repo_text
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