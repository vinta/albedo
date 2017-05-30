package ws.vinta.albedo.preprocessors

import java.util.UUID

import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.functions._

class RatingBuilder extends Transformer {
  override val uid: String = UUID.randomUUID().toString()

  override def transformSchema(schema: StructType): StructType = {
    schema
  }

  override def transform(dataset: Dataset[_]): DataFrame = {
    dataset
      .selectExpr("from_user_id AS user", "repo_id AS item", "1 AS rating", "starred_at")
      .orderBy(col("user"), desc("starred_at"))
  }

  override def copy(extra: ParamMap): Transformer = {
    this
  }
}