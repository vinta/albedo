package ws.vinta.albedo.preprocessors

import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.{DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.functions.{lower, when}
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset}

class RepoInfoCleaner(override val uid: String)
  extends Transformer with DefaultParamsWritable {

  def this() {
    this(Identifiable.randomUID("userInfoTransformer"))
  }

  override def transformSchema(schema: StructType): StructType = {
    schema
  }

  override def transform(dataset: Dataset[_]): DataFrame = {
    import dataset.sparkSession.implicits._

    dataset
      .where($"stargazers_count" >= 2)
      .na.fill("", Array("description", "homepage"))
      .withColumn("description", lower($"description"))
      .withColumn("full_name", lower($"full_name"))
      .withColumn("homepage", lower($"homepage"))
      .withColumn("topics", lower($"topics"))
      .withColumn("is_unmaintained", when($"description".like("%unmaintained%") or
                                          $"description".like("%no longer maintained%") or
                                          $"description".like("%no longer actively maintained%") or
                                          $"description".like("%not maintained%") or
                                          $"description".like("%not actively maintained%"), 1).otherwise(0))
      .where($"is_unmaintained" === 0)
      .drop($"is_unmaintained")
  }

  override def copy(extra: ParamMap): this.type = {
    defaultCopy(extra)
  }
}