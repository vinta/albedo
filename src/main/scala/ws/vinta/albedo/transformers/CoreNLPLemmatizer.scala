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
    def equalsIgnoreNullability(left: DataType, right: DataType): Boolean = {
      (left, right) match {
        case (ArrayType(leftElementType, _), ArrayType(rightElementType, _)) =>
          equalsIgnoreNullability(leftElementType, rightElementType)
        case (MapType(leftKeyType, leftValueType, _), MapType(rightKeyType, rightValueType, _)) =>
          equalsIgnoreNullability(leftKeyType, rightKeyType) && equalsIgnoreNullability(leftValueType, rightValueType)
        case (StructType(leftFields), StructType(rightFields)) =>
          leftFields.length == rightFields.length && leftFields.zip(rightFields).forall { case (l, r) =>
            l.name == r.name && equalsIgnoreNullability(l.dataType, r.dataType)
          }
        case (l, r) => l == r
      }
    }

    val expectedDataType = ArrayType(StringType, true)
    require(equalsIgnoreNullability(inputType, expectedDataType), s"Input type must be $expectedDataType but got $inputType.")
  }

  override protected def outputDataType: DataType = {
    ArrayType(StringType, true)
  }

  override def copy(extra: ParamMap): this.type = {
    defaultCopy(extra)
  }
}

object CoreNLPLemmatizer extends DefaultParamsReadable[CoreNLPLemmatizer]