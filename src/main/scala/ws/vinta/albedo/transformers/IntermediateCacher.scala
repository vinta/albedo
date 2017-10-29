package ws.vinta.albedo.transformers

import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.{ParamMap, StringArrayParam}
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Dataset}

class IntermediateCacher(override val uid: String)
  extends Transformer with DefaultParamsWritable {

  def this() = {
    this(Identifiable.randomUID("intermediateCacher"))
  }

  val inputCols = new StringArrayParam(this, "inputCols", "Input column names")

  def getInputCols: Array[String] = $(inputCols)

  def setInputCols(value: Array[String]): this.type = set(inputCols, value)
  setDefault(inputCols -> Array.empty[String])

  override def transformSchema(schema: StructType): StructType = {
    schema
  }

  override def transform(dataset: Dataset[_]): DataFrame = {
    transformSchema(dataset.schema)

    val intermediateDF = if ($(inputCols).isEmpty) dataset.toDF() else dataset.select($(inputCols).map(col(_)): _*)
    intermediateDF.cache()
  }

  override def copy(extra: ParamMap): IntermediateCacher = {
    defaultCopy(extra)
  }
}

object IntermediateCacher extends DefaultParamsReadable[IntermediateCacher]