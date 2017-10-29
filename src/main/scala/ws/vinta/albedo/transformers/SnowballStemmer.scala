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

  override def createTransformFunc: Seq[String] => Seq[String] = { strings =>
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

  override def validateInputType(inputType: DataType): Unit = {
    require(inputType == ArrayType(StringType), s"Input type must be string type but got $inputType.")
  }

  override def outputDataType: DataType = {
    ArrayType(StringType)
  }

  override def copy(extra: ParamMap): SnowballStemmer = {
    defaultCopy(extra)
  }
}

object SnowballStemmer extends DefaultParamsReadable[SnowballStemmer]