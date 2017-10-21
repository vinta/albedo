package ws.vinta.albedo.transformers

import org.apache.spark.ml.UnaryTransformer
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.types.{ArrayType, DataType, StringType}
import org.tartarus.snowball.ext.EnglishStemmer

class SnowballStemmer(override val uid: String)
  extends UnaryTransformer[Seq[String], Seq[String], SnowballStemmer] with DefaultParamsWritable {

  def this() = {
    this(Identifiable.randomUID("snowballStemmer"))
  }

  override protected def createTransformFunc: Seq[String] => Seq[String] = { strings =>
    val stemmer = new EnglishStemmer()

    strings.map((str: String) => {
      try {
        stemmer.setCurrent(str)
        stemmer.stem()
        stemmer.getCurrent()
      } catch {
        case _: Exception => str
      }
    })
  }

  override protected def validateInputType(inputType: DataType): Unit = {
    require(inputType == ArrayType(StringType), s"Input type must be string type but got $inputType.")
  }

  override protected def outputDataType: DataType = {
    ArrayType(StringType)
  }

  override def copy(extra: ParamMap): this.type = {
    defaultCopy(extra)
  }
}

object SnowballStemmer extends DefaultParamsReadable[SnowballStemmer]