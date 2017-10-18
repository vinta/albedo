package ws.vinta.albedo

import org.apache.spark.SparkConf
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, SparkSession}
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

    import spark.implicits._

    // Load Data

    val rawRepoInfoDS = loadRawRepoInfoDS()
    rawRepoInfoDS.cache()

    val rawUserInfoDS = loadRawUserInfoDS()
    rawUserInfoDS.cache()

    val rawStarringDS = loadRawStarringDS()
    rawStarringDS.cache()

    // Feature Engineering

    val continuousColumnNames = mutable.ArrayBuffer.empty[String]
    val categoricalColumnNames = mutable.ArrayBuffer.empty[String]
    val textColumnNames = mutable.ArrayBuffer.empty[String]

    continuousColumnNames += "stargazers_count"
    continuousColumnNames += "forks_count"
    continuousColumnNames += "subscribers_count"
    continuousColumnNames += "open_issues_count"

    // TODO: listColumnNames: topics

    // Impute Data

    val nullableColumnNames = Array("description", "homepage")

    val imputedRepoInfoDF = rawRepoInfoDS
      .where($"stargazers_count".between(2, 100000))
      .where($"forks_count" <= 90000)
      .where($"fork" === false)
      .na.fill("", nullableColumnNames)

    // Clean Data

    val lowerableColumnNames = Array("description", "language", "topics")

    val booleanColumnNames = Array("has_issues", "has_projects", "has_downloads", "has_wiki", "has_pages")

    val cleanRepoInfoDF = (lowerableColumnNames ++ booleanColumnNames).foldLeft[DataFrame](imputedRepoInfoDF)((accDF, columnName) => {
      columnName match {
        case _ if lowerableColumnNames.contains(columnName) =>
          accDF.withColumn(s"clean_$columnName", lower(col(columnName)))
        case _ if booleanColumnNames.contains(columnName) =>
          accDF.withColumn(s"clean_$columnName", col(columnName).cast("double"))
      }
    })
    cleanRepoInfoDF.cache()

    categoricalColumnNames += "owner_type"
    categoricalColumnNames.append(booleanColumnNames.map(columnName => s"clean_$columnName"): _*)

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
      .withColumn("days_between_created_at_today", datediff(current_date(), $"created_at"))
      .withColumn("days_between_updated_at_today", datediff(current_date(), $"updated_at"))
      .withColumn("days_between_pushed_at_today", datediff(current_date(), $"pushed_at"))
      .withColumn("watch_star_count_ratio", round($"subscribers_count" / ($"stargazers_count" + lit(1)), 3))
      .withColumn("fork_star_count_ratio", round($"forks_count" / ($"stargazers_count" + lit(1)), 3))
      .withColumn("is_vinta_starred", when($"repo_id".isin(vintaStarredRepos: _*), 1.0).otherwise(0.0))
      .withColumn("text", concat_ws(" ", $"owner_username", $"name", $"language", $"description", $"topics"))

    continuousColumnNames += "days_between_created_at_today"
    continuousColumnNames += "days_between_updated_at_today"
    continuousColumnNames += "days_between_pushed_at_today"
    continuousColumnNames += "watch_star_count_ratio"
    continuousColumnNames += "fork_star_count_ratio"

    categoricalColumnNames += "is_vinta_starred"

    textColumnNames += "text"

    // Transform Features

    val languagesDF = cleanRepoInfoDF
      .groupBy($"clean_language")
      .agg(count("*").alias("count_per_language"))

    val transformedRepoInfoDF = constructedRepoInfoDF
      .join(languagesDF, Seq("clean_language"))
      .withColumn("has_homepage", when($"homepage" === "", 0.0).otherwise(1.0))
      .withColumn("binned_language", when($"count_per_language" <= 30, "__other").otherwise($"clean_language"))
    transformedRepoInfoDF.cache()

    categoricalColumnNames += "has_homepage"
    categoricalColumnNames += "binned_language"

    // Save Results

    println("Continuous column names: " + continuousColumnNames.mkString(", "))
    println("Categorical column names: " + categoricalColumnNames.mkString(", "))
    println("Text column names: " + textColumnNames.mkString(", "))

    val featureNames = mutable.ArrayBuffer("repo_id", "full_name") ++ continuousColumnNames ++ categoricalColumnNames ++ textColumnNames
    val path = s"${settings.dataDir}/${settings.today}/repoProfileDF.parquet"
    val repoProfileDF = loadOrCreateDataFrame(path, () => {
      transformedRepoInfoDF.select(featureNames.map(col): _*)
    })

    repoProfileDF.show(false)

    spark.stop()
  }
}