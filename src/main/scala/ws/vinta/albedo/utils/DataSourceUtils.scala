package ws.vinta.albedo.utils

import java.util.Properties

import org.apache.spark.sql.functions.lit
import org.apache.spark.sql.{AnalysisException, DataFrame, Dataset, SparkSession}
import ws.vinta.albedo.schemas.{RepoInfo, RepoStarring}

object DataSourceUtils {
  private val dbUrl = "jdbc:mysql://127.0.0.1:3306/albedo?verifyServerCertificate=false&useSSL=false&rewriteBatchedStatements=true"
  private val props = new Properties()
  props.setProperty("driver", "com.mysql.jdbc.Driver")
  props.setProperty("user", "root")
  props.setProperty("password", "123")

  private val dateFormatter = new java.text.SimpleDateFormat("yyyyMMdd")
  private val today = dateFormatter.format(new java.util.Date())

  def loadRepoInfo()(implicit spark: SparkSession): Dataset[RepoInfo] = {
    import spark.implicits._

    val savePath = s"spark-data/$today/repoInfoDS.parquet"
    val repoInfoDF: DataFrame = try {
      spark.read.parquet(savePath)
    } catch {
      case e: AnalysisException => {
        if (e.getMessage().contains("Path does not exist")) {
          val tempRepoInfoDF = spark.read.jdbc(dbUrl, "app_repoinfo", props)
          tempRepoInfoDF.write.parquet(savePath)
          tempRepoInfoDF
        } else {
          throw e
        }
      }
    }
    repoInfoDF.as[RepoInfo]
  }

  def loadRepoStarring()(implicit spark: SparkSession): Dataset[RepoStarring] = {
    import spark.implicits._

    val savePath = s"spark-data/$today/repoStarringDS.parquet"
    val repoStarringDF: DataFrame = try {
      spark.read.parquet(savePath)
    } catch {
      case e: AnalysisException => {
        if (e.getMessage().contains("Path does not exist")) {
          var tempRepoStarringDF = spark.read.jdbc(dbUrl, "app_repostarring", props)
          tempRepoStarringDF = tempRepoStarringDF.withColumn("starring", lit(1))
          tempRepoStarringDF.write.parquet(savePath)
          tempRepoStarringDF
        } else {
          throw e
        }
      }
    }
    repoStarringDF.as[RepoStarring]
  }
}