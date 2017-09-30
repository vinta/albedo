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

    //val path = s"${settings.dataDir}/${settings.today}/rawUserInfoDF.parquet"
    val path = s"${settings.dataDir}/20170903/rawUserInfoDF.parquet"
    val df = loadOrCreateDataFrame(path, () => {
      spark.read.jdbc(dbUrl, "app_userinfo", props).withColumnRenamed("id", "user_id")
    })
    df.as[UserInfo]
  }

  def loadRawRepoInfoDS()(implicit spark: SparkSession): Dataset[RepoInfo] = {
    import spark.implicits._

    //val path = s"${settings.dataDir}/${settings.today}/rawRepoInfoDF.parquet"
    val path = s"${settings.dataDir}/20170903/rawRepoInfoDF.parquet"
    val df = loadOrCreateDataFrame(path, () => {
      spark.read.jdbc(dbUrl, "app_repoinfo", props).withColumnRenamed("id", "repo_id")
    })
    df.as[RepoInfo]
  }

  def loadRawStarringDS()(implicit spark: SparkSession): Dataset[Starring] = {
    import spark.implicits._

    //val path = s"${settings.dataDir}/${settings.today}/rawStarringDF.parquet"
    val path = s"${settings.dataDir}/20170903/rawStarringDF.parquet"
    val df = loadOrCreateDataFrame(path, () => {
      spark.read.jdbc(dbUrl, "app_repostarring", props)
        .select($"user_id", $"repo_id", $"starred_at")
        .withColumn("starring", lit(1.0))
    })
    df.as[Starring]
  }

  def loadRawRelationDS()(implicit spark: SparkSession): Dataset[Relation] = {
    import spark.implicits._

    //val path = s"${settings.dataDir}/${settings.today}/rawRelationDF.parquet"
    val path = s"${settings.dataDir}/20170903/rawRelationDF.parquet"
    val df = loadOrCreateDataFrame(path, () => {
      spark.read.jdbc(dbUrl, "app_userrelation", props).select($"from_user_id", $"to_user_id", $"relation")
    })
    df.as[Relation]
  }

  def loadUserProfileDF()(implicit spark: SparkSession): DataFrame = {
    //val path = s"${settings.dataDir}/${settings.today}/userProfileDF.parquet"
    val path = s"${settings.dataDir}/20170903/userProfileDF.parquet"
    spark.read.parquet(path)
  }

  def loadRepoProfileDF()(implicit spark: SparkSession): DataFrame = {
    //val path = s"${settings.dataDir}/${settings.today}/repoProfileDF.parquet"
    val path = s"${settings.dataDir}/20170903/repoProfileDF.parquet"
    spark.read.parquet(path)
  }

  def loadPopularRepoDF()(implicit spark: SparkSession): DataFrame = {
    import spark.implicits._

    //val path = s"${settings.dataDir}/${settings.today}/popularRepoDF.parquet"
    val path = s"${settings.dataDir}/20170903/popularRepoDF.parquet"
    val df = loadOrCreateDataFrame(path, () => {
      val rawRepoInfoDS = loadRawRepoInfoDS()
      val df = rawRepoInfoDS
        .select($"repo_id", $"stargazers_count")
        .where($"stargazers_count" >= 1000)
        .orderBy($"stargazers_count".desc)
      df
    })
    df
  }
}
