package ws.vinta.albedo.utils

import java.util.Properties

import org.apache.spark.sql.functions._
import org.apache.spark.sql.{AnalysisException, DataFrame, Dataset, SparkSession}
import ws.vinta.albedo.schemas._
import ws.vinta.albedo.settings

object DatasetUtils {
  private val dbUrl = "jdbc:mysql://127.0.0.1:3306/albedo?verifyServerCertificate=false&useSSL=false&rewriteBatchedStatements=true"
  private val props = new Properties()
  props.setProperty("driver", "com.mysql.jdbc.Driver")
  props.setProperty("user", "root")
  props.setProperty("password", "123")

  def randomSplitByUser(df: DataFrame, userCol: String, trainingTestRatio: Double): Array[DataFrame] = {
    val spark = df.sparkSession
    import spark.implicits._

    val fractions = df
      .select(userCol)
      .distinct()
      .map(row => (row.getInt(0), trainingTestRatio))
      .collect()
      .to[List]
      .toMap
    val trainingDF = df.stat.sampleBy("user_id", fractions, 42)

    val testRDD = df.rdd.subtract(trainingDF.rdd)
    val testDF = spark.createDataFrame(testRDD, df.schema)

    Array(trainingDF, testDF)
  }

  def loadOrCreateDataFrame(path: String, createDataFrameFunc: () => DataFrame)(implicit spark: SparkSession): DataFrame = {
    try {
      spark.read.parquet(path)
    } catch {
      case e: AnalysisException => {
        if (e.getMessage().contains("Path does not exist")) {
          val df = createDataFrameFunc()
          df.write.mode("overwrite").parquet(path)
          df
        } else {
          throw e
        }
      }
    }
  }

  def loadRawUserInfoDS()(implicit spark: SparkSession): Dataset[UserInfo] = {
    import spark.implicits._

    val path = s"${settings.dataDir}/${settings.today}/rawUserInfoDF.parquet"
    val df = loadOrCreateDataFrame(path, () => {
      spark.read.jdbc(dbUrl, "app_userinfo", props)
        .withColumnRenamed("id", "user_id")
        .withColumnRenamed("login", "user_login")
        .withColumnRenamed("account_type", "user_account_type")
        .withColumnRenamed("name", "user_name")
        .withColumnRenamed("company", "user_company")
        .withColumnRenamed("blog", "user_blog")
        .withColumnRenamed("location", "user_location")
        .withColumnRenamed("email", "user_email")
        .withColumnRenamed("bio", "user_bio")
        .withColumnRenamed("public_repos", "user_public_repos_count")
        .withColumnRenamed("public_gists", "user_public_gists_count")
        .withColumnRenamed("followers", "user_followers_count")
        .withColumnRenamed("following", "user_following_count")
        .withColumnRenamed("created_at", "user_created_at")
        .withColumnRenamed("updated_at", "user_updated_at")
    })
    df.as[UserInfo]
  }

  def loadRawRepoInfoDS()(implicit spark: SparkSession): Dataset[RepoInfo] = {
    import spark.implicits._

    val path = s"${settings.dataDir}/${settings.today}/rawRepoInfoDF.parquet"
    val df = loadOrCreateDataFrame(path, () => {
      spark.read.jdbc(dbUrl, "app_repoinfo", props)
        .withColumnRenamed("id", "repo_id")
        .withColumnRenamed("owner_id", "repo_owner_id")
        .withColumnRenamed("owner_username", "repo_owner_username")
        .withColumnRenamed("owner_type", "repo_owner_type")
        .withColumnRenamed("name", "repo_name")
        .withColumnRenamed("full_name", "repo_full_name")
        .withColumnRenamed("description", "repo_description")
        .withColumnRenamed("language", "repo_language")
        .withColumnRenamed("created_at", "repo_created_at")
        .withColumnRenamed("updated_at", "repo_updated_at")
        .withColumnRenamed("pushed_at", "repo_pushed_at")
        .withColumnRenamed("homepage", "repo_homepage")
        .withColumnRenamed("size", "repo_size")
        .withColumnRenamed("stargazers_count", "repo_stargazers_count")
        .withColumnRenamed("forks_count", "repo_forks_count")
        .withColumnRenamed("subscribers_count", "repo_subscribers_count")
        .withColumnRenamed("fork", "repo_is_fork")
        .withColumnRenamed("has_issues", "repo_has_issues")
        .withColumnRenamed("has_projects", "repo_has_projects")
        .withColumnRenamed("has_downloads", "repo_has_downloads")
        .withColumnRenamed("has_wiki", "repo_has_wiki")
        .withColumnRenamed("has_pages", "repo_has_pages")
        .withColumnRenamed("open_issues_count", "repo_open_issues_count")
        .withColumnRenamed("topics", "repo_topics")
    })
    df.as[RepoInfo]
  }

  def loadRawStarringDS()(implicit spark: SparkSession): Dataset[Starring] = {
    import spark.implicits._

    val path = s"${settings.dataDir}/${settings.today}/rawStarringDF.parquet"
    val df = loadOrCreateDataFrame(path, () => {
      spark.read.jdbc(dbUrl, "app_repostarring", props)
        .select($"user_id", $"repo_id", $"starred_at")
        .withColumn("starring", lit(1.0))
    })
    df
      .repartition($"user_id")
      .as[Starring]
  }

  def loadRawRelationDS()(implicit spark: SparkSession): Dataset[Relation] = {
    import spark.implicits._

    val path = s"${settings.dataDir}/${settings.today}/rawRelationDF.parquet"
    val df = loadOrCreateDataFrame(path, () => {
      spark.read.jdbc(dbUrl, "app_userrelation", props)
        .select($"from_user_id", $"to_user_id", $"relation")
    })
    df
      .repartition($"from_user_id")
      .as[Relation]
  }

  def loadUserProfileDF()(implicit spark: SparkSession): DataFrame = {
    val path = s"${settings.dataDir}/${settings.today}/userProfileDF.parquet"
    spark.read.parquet(path)
  }

  def loadRepoProfileDF()(implicit spark: SparkSession): DataFrame = {
    val path = s"${settings.dataDir}/${settings.today}/repoProfileDF.parquet"
    spark.read.parquet(path)
  }

  def loadPopularRepoDF()(implicit spark: SparkSession): DataFrame = {
    import spark.implicits._

    val path = s"${settings.dataDir}/${settings.today}/popularRepoDF.parquet"
    loadOrCreateDataFrame(path, () => {
      val rawRepoInfoDS = loadRawRepoInfoDS()
      rawRepoInfoDS
        .select($"repo_id", $"repo_stargazers_count", $"repo_created_at")
        .where($"repo_stargazers_count".between(1000, 290000))
        .orderBy($"repo_stargazers_count".desc)
        .repartition(1)
    })
  }
}