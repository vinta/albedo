package ws.vinta.albedo

import org.apache.spark.sql.SparkSession

object RepoProfileBuilder {
  def main(args: Array[String]): Unit = {
    implicit val spark: SparkSession = SparkSession
      .builder()
      .appName("GitHubCorpusTrainer")
      .getOrCreate()

    spark.stop()
  }
}