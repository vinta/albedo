package ws.vinta.albedo.recommenders

import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}
import ws.vinta.albedo.utils.DatasetUtils._

class CurationRecommender(override val uid: String) extends Recommender {

  def this() = {
    this(Identifiable.randomUID("curationRecommender"))
  }

  override def source = "curation"

  override def recommendForUsers(userDF: Dataset[_]): DataFrame = {
    transformSchema(userDF.schema)

    implicit val spark: SparkSession = userDF.sparkSession
    import spark.implicits._

    val rawStarringDS = loadRawStarringDS().cache()

    val curatorIds = Array(652070, 1912583, 59990, 646843, 28702) // vinta, saiday, tzangms, fukuball, wancw
    val curatedRepoDF = rawStarringDS
      .select($"repo_id", $"starred_at")
      .where($"user_id".isin(curatorIds: _*))
      .groupBy($"repo_id")
      .agg(max($"starred_at").alias("starred_at"))
      .orderBy($"starred_at".desc)
      .limit($(topK))
      .cache()

    def calculateScoreUDF = udf((starred_at: java.sql.Timestamp) => {
      starred_at.getTime / 1000.0
    })

    userDF
      .select($(userCol))
      .crossJoin(curatedRepoDF)
      .select(col($(userCol)), $"repo_id".alias($(itemCol)), calculateScoreUDF($"starred_at").alias($(scoreCol)))
      .withColumn($(sourceCol), lit(source))
  }
}