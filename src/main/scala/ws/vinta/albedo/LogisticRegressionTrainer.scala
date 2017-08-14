package ws.vinta.albedo

import org.apache.spark.sql.{Dataset, SparkSession}
import ws.vinta.albedo.schemas.{RepoInfo, RepoStarring}
import ws.vinta.albedo.utils.DataSourceUtils.{loadRepoInfo, loadRepoStarring}

object LogisticRegressionTrainer {
  val appName = "LogisticRegressionTrainer"

  def main(args: Array[String]): Unit = {
    val activeUser = args(1)
    println(activeUser)

    implicit val spark = SparkSession
      .builder()
      .appName(appName)
      .getOrCreate()

    implicit val sc = spark.sparkContext
    sc.setLogLevel("WARN")

    import spark.implicits._

    // load data

    val repoStarringDS: Dataset[RepoStarring] = loadRepoStarring()
    repoStarringDS.cache()
    repoStarringDS.show()

    val repoInfoDS: Dataset[RepoInfo] = loadRepoInfo()
    repoInfoDS.cache()
    repoInfoDS.show()

    spark.stop()
  }
}