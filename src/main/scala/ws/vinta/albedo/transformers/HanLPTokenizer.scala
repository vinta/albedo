package ws.vinta.albedo.transformers

import java.util

import com.hankcs.hanlp.HanLP
import com.hankcs.hanlp.dictionary.stopword.CoreStopWordDictionary
import com.hankcs.hanlp.seg.common.Term
import org.apache.spark.ml.UnaryTransformer
import org.apache.spark.ml.param.{BooleanParam, ParamMap}
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.types._

import scala.collection.JavaConverters._

class HanLPTokenizer(override val uid: String)
  extends UnaryTransformer[String, Seq[String], HanLPTokenizer] with DefaultParamsWritable {

  def this() = {
    this(Identifiable.randomUID("hanLPTokenizer"))
  }

  val shouldRemoveStopWords = new BooleanParam(this, "shouldRemoveStopWords", "Whether to remove stop words")

  def getShouldRemoveStopWords: Boolean = $(shouldRemoveStopWords)

  def setShouldRemoveStopWords(value: Boolean): this.type = set(shouldRemoveStopWords, value)
  setDefault(shouldRemoveStopWords -> true)

  override protected def createTransformFunc: String => Seq[String] = { originStr =>
    HanLP.Config.ShowTermNature = false
    HanLP.Config.Normalization = false
    val segment = HanLP.newSegment()
    val termList: util.List[Term] = segment.seg(HanLP.convertToSimplifiedChinese(originStr.toLowerCase))

    if ($(shouldRemoveStopWords)) {
      CoreStopWordDictionary.apply(termList)
    }

    val LanguageRE = """(c|r|c\+\+|c#|f#)""".r
    val OneCharExceptCJKRE = """([^\p{InHiragana}\p{InKatakana}\p{InBopomofo}\p{InCJKCompatibilityIdeographs}\p{InCJKUnifiedIdeographs}])""".r
    termList
      .asScala
      .flatMap((term: Term) => {
        val word = term.word
        word match {
          case LanguageRE(language) => Array(language)
          case OneCharExceptCJKRE(_) => Array.empty[String]
          case _ => """([\w\.\-_\p{InHiragana}\p{InKatakana}\p{InBopomofo}\p{InCJKCompatibilityIdeographs}\p{InCJKUnifiedIdeographs}]+)""".r.findAllIn(word).toArray
        }
      })
  }

  override protected def validateInputType(inputType: DataType): Unit = {
    require(inputType == StringType, s"Input type must be string type but got $inputType.")
  }

  override protected def outputDataType: DataType = {
    new ArrayType(StringType, false)
  }

  override def copy(extra: ParamMap): this.type = {
    defaultCopy(extra)
  }
}

object HanLPTokenizer extends DefaultParamsReadable[HanLPTokenizer]