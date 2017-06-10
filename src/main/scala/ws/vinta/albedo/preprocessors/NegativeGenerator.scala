package ws.vinta.albedo.preprocessors

import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.sql.types.StructType

class NegativeGenerator(override val uid: String) extends Transformer {
  def this() = this(Identifiable.randomUID("NegativeGenerator"))

  override def transformSchema(schema: StructType): StructType = {
    schema
  }

  override def transform(dataset: Dataset[_]): DataFrame = {
    dataset.toDF()
  }

  override def copy(extra: ParamMap): NegativeGenerator = {
    defaultCopy(extra)
  }
}