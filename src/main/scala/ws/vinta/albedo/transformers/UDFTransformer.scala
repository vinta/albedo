package ws.vinta.albedo.transformers

import org.apache.spark.ml.UnaryTransformer
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.types._

class UDFTransformer(override val uid: String)
  extends UnaryTransformer[String, Seq[String], UDFTransformer] with DefaultParamsWritable {

  def this() = {
    this(Identifiable.randomUID("udfTransformer"))
  }

  override protected def createTransformFunc: String => Seq[String] = { originStr =>
    Seq("abc")
  }

  override protected def validateInputType(inputType: DataType): Unit = {
    require(inputType == StringType, s"Input type must be string type but got $inputType.")
  }

  override protected def outputDataType: DataType = {
    ArrayType(StringType)
  }

  override def copy(extra: ParamMap): this.type = {
    defaultCopy(extra)
  }
}

object UDFTransformer extends DefaultParamsReadable[UDFTransformer]