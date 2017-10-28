package ws.vinta.albedo.transformers

import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.{Param, ParamMap, StringArrayParam}
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.types._

abstract class SimpleTransformer(override val uid: String)
  extends Transformer with DefaultParamsWritable {

  def this() = {
    this(Identifiable.randomUID("simpleTransformer"))
  }

  val inputCols: StringArrayParam = new StringArrayParam(this, "inputCols", "Input column names")

  def getInputCols: Array[String] = $(inputCols)

  def setInputCols(value: Array[String]): this.type = set(inputCols, value)

  val outputCol: Param[String] = new Param[String](this, "outputCol", "Output column name")

  def getOutputCol: String = $(outputCol)

  def setOutputCol(value: String): this.type = set(outputCol, value)
  setDefault(outputCol, s"${uid}_output")

  def outputDataType: DataType = ???

  override def transform(dataset: Dataset[_]) = ???

  override def transformSchema(schema: StructType) = {
    $(inputCols).foreach((inputColName: String) => {
      require(!schema.fieldNames.contains(inputColName), s"Input column $inputColName must be exist.")
    })

    require(schema.fieldNames.contains($(outputCol)), s"Output column ${$(outputCol)} already exists.")

    val outputFields = schema.fields :+ StructField($(outputCol), outputDataType, nullable = false)
    StructType(outputFields)
  }

  override def copy(extra: ParamMap): this.type = {
    defaultCopy(extra)
  }
}

object SimpleTransformer extends DefaultParamsReadable[SimpleTransformer]