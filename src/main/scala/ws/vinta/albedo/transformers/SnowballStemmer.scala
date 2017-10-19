package ws.vinta.albedo.transformers

import org.apache.spark.ml.UnaryTransformer
import org.apache.spark.ml.param.{Param, ParamMap}
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.types.{ArrayType, DataType, StringType}
import org.tartarus.snowball.{SnowballStemmer => TartarusSnowballStemmer}

class SnowballStemmer(override val uid: String)
  extends UnaryTransformer[Seq[String], Seq[String], SnowballStemmer] with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("snowballStemmer"))

  val language: Param[String] = new Param(this, "language", "Stemming language")

  def getLanguage: String = $(language)

  def setLanguage(value: String): this.type = set(language, value)
  setDefault(language -> "english")

  override protected def createTransformFunc: Seq[String] => Seq[String] = { strings =>
    val stemmerClass = Class.forName("org.tartarus.snowball.ext." + s"${$(language).toLowerCase}Stemmer")
    val stemmer = stemmerClass.newInstance().asInstanceOf[TartarusSnowballStemmer]
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

  override def copy(extra: ParamMap): this.type = defaultCopy(extra)
}

object SnowballStemmer extends DefaultParamsReadable[SnowballStemmer]