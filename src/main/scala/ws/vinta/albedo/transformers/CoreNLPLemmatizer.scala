package ws.vinta.albedo.transformers

import edu.stanford.nlp.simple.Sentence
import org.apache.spark.ml.UnaryTransformer
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.types._

class CoreNLPLemmatizer(override val uid: String)
  extends UnaryTransformer[Seq[String], Seq[String], CoreNLPLemmatizer] with DefaultParamsWritable {

  def this() = {
    this(Identifiable.randomUID("coreNLPLemmatizer"))
  }

  override protected def createTransformFunc: Seq[String] => Seq[String] = { words =>
    words.map((word: String) => {
      val sentence = new Sentence(word)
      val lemmatized = sentence.lemmas()
      lemmatized.toArray.mkString(" ")
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

object CoreNLPLemmatizer extends DefaultParamsReadable[CoreNLPLemmatizer]