package ws.vinta.albedo.utils

import java.util.Properties

import org.apache.spark.sql.functions.lit
import org.apache.spark.sql.{AnalysisException, DataFrame, Dataset, SparkSession}
import ws.vinta.albedo.schemas._
import ws.vinta.albedo.settings

object DatasetUtils {
  private val dbUrl = "jdbc:mysql://127.0.0.1:3306/albedo?verifyServerCertificate=false&useSSL=false&rewriteBatchedStatements=true"
  private val props = new Properties()
  props.setProperty("driver", "com.mysql.jdbc.Driver")
  props.setProperty("user", "root")
  props.setProperty("password", "123")

  def loadRawUserInfoDS()(implicit spark: SparkSession): Dataset[UserInfo] = {
    import spark.implicits._

    //val savePath = s"${settings.dataDir}/${settings.today}/rawUserInfoDF.parquet"
    val savePath = s"${settings.dataDir}/20170903/rawUserInfoDF.parquet"
    val df = try {
      spark.read.parquet(savePath)
    } catch {
      case e: AnalysisException => {
        if (e.getMessage().contains("Path does not exist")) {
          val df = spark.read.jdbc(dbUrl, "app_userinfo", props).withColumnRenamed("id", "user_id")
          df.write.mode("overwrite").parquet(savePath)
          df
        } else {
          throw e
        }
      }
    }

    df.as[UserInfo]
  }

  def loadRawUserRelationDS()(implicit spark: SparkSession): Dataset[UserRelation] = {
    import spark.implicits._

    //val savePath = s"${settings.dataDir}/${settings.today}/rawUserRelationDF.parquet"
    val savePath = s"${settings.dataDir}/20170903/rawUserRelationDF.parquet"
    val df = try {
      spark.read.parquet(savePath)
    } catch {
      case e: AnalysisException => {
        if (e.getMessage().contains("Path does not exist")) {
          val df = spark.read.jdbc(dbUrl, "app_userrelation", props).select($"from_user_id", $"to_user_id", $"relation")
          df.write.mode("overwrite").parquet(savePath)
          df
        } else {
          throw e
        }
      }
    }

    df.as[UserRelation]
  }

  def loadRawRepoInfoDS()(implicit spark: SparkSession): Dataset[RepoInfo] = {
    import spark.implicits._

    //val savePath = s"${settings.dataDir}/${settings.today}/rawRepoInfoDF.parquet"
    val savePath = s"${settings.dataDir}/20170903/rawRepoInfoDF.parquet"
    val df = try {
      spark.read.parquet(savePath)
    } catch {
      case e: AnalysisException => {
        if (e.getMessage().contains("Path does not exist")) {
          val df = spark.read.jdbc(dbUrl, "app_repoinfo", props).withColumnRenamed("id", "repo_id")
          df.write.mode("overwrite").parquet(savePath)
          df
        } else {
          throw e
        }
      }
    }

    df.as[RepoInfo]
  }

  def loadRawRepoStarringDS()(implicit spark: SparkSession): Dataset[RepoStarring] = {
    import spark.implicits._

    //val savePath = s"${settings.dataDir}/${settings.today}/rawRepoStarringDF.parquet"
    val savePath = s"${settings.dataDir}/20170903/rawRepoStarringDF.parquet"
    val df = try {
      spark.read.parquet(savePath)
    } catch {
      case e: AnalysisException => {
        if (e.getMessage().contains("Path does not exist")) {
          val df = spark.read.jdbc(dbUrl, "app_repostarring", props)
            .select($"user_id", $"repo_id", $"starred_at")
            .withColumn("starring", lit(1.0))
          df.write.mode("overwrite").parquet(savePath)
          df
        } else {
          throw e
        }
      }
    }

    df.as[RepoStarring]
  }

  def loadUserProfileDF()(implicit spark: SparkSession): DataFrame = {
    //val savePath = s"${settings.dataDir}/${settings.today}/userProfileDF.parquet"
    val savePath = s"${settings.dataDir}/20170903/userProfileDF.parquet"
    spark.read.parquet(savePath)
  }

  def loadRepoProfileDF()(implicit spark: SparkSession): DataFrame = {
    //val savePath = s"${settings.dataDir}/${settings.today}/repoProfileDF.parquet"
    val savePath = s"${settings.dataDir}/20170903/repoProfileDF.parquet"
    spark.read.parquet(savePath)
  }

  def loadPopularRepoDF()(implicit spark: SparkSession): DataFrame = {
    import spark.implicits._

    //val savePath = s"${settings.dataDir}/${settings.today}/popularRepoDF.parquet"
    val savePath = s"${settings.dataDir}/20170903/popularRepoDF.parquet"
    def df: DataFrame = try {
      spark.read.parquet(savePath)
    } catch {
      case e: AnalysisException => {
        if (e.getMessage().contains("Path does not exist")) {
          val rawRepoInfoDS = loadRawRepoInfoDS()
          val df = rawRepoInfoDS
            .select($"repo_id", $"stargazers_count")
            .where($"stargazers_count" >= 1000)
            .orderBy($"stargazers_count".desc)
          df.write.mode("overwrite").parquet(savePath)
          df
        } else {
          throw e
        }
      }
    }
    df
  }
}