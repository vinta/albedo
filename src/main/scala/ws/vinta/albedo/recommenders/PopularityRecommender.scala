package ws.vinta.albedo.recommenders

import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}
import ws.vinta.albedo.utils.DatasetUtils._

class PopularityRecommender(override val uid: String) extends Recommender {

  def this() = {
    this(Identifiable.randomUID("popularityRecommender"))
  }

  override def source = "popularity"

  override def recommendForUsers(userDF: Dataset[_]): DataFrame = {
    transformSchema(userDF.schema)

    implicit val spark: SparkSession = userDF.sparkSession
    import spark.implicits._

    val popularRepoDF = loadPopularRepoDF().limit($(topK))
    popularRepoDF.cache()

    def calculateScoreUDF = udf((stargazers_count: Int, created_at: java.sql.Timestamp) => {
      val valueScore = math.round(math.log10(stargazers_count) * 1000.0) / 1000.0
      val timeScore = (created_at.getTime / 1000.0) / (60 * 60 * 24 * 30 * 12) / 5.0
      valueScore + timeScore
    })

    userDF
      .select($(userCol))
      .crossJoin(popularRepoDF)
      .select(col($(userCol)), $"repo_id".alias($(itemCol)), calculateScoreUDF($"stargazers_count", $"created_at").alias($(scoreCol)))
      .withColumn($(sourceCol), lit(source))
  }
}