package ws.vinta.albedo.transformers

import com.hankcs.hanlp.HanLP
import com.hankcs.hanlp.dictionary.stopword.CoreStopWordDictionary
import com.hankcs.hanlp.seg.common.Term
import org.apache.spark.ml.UnaryTransformer
import org.apache.spark.ml.param.{Param, ParamMap}
import org.apache.spark.ml.util.{DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.types.{ArrayType, DataType, StringType}

class HanLPTokenizer(override val uid: String)
  extends UnaryTransformer[String, Seq[String], HanLPTokenizer] with DefaultParamsWritable {

  def this() = {
    this(Identifiable.randomUID("hanLPTokenizer"))
  }

  val shouldRemoveStopWords = new Param[Boolean](this, "shouldRemoveStopWords", "是否移除 Stop Words")

  def getShouldRemoveStopWords: Boolean = $(shouldRemoveStopWords)

  def setShouldRemoveStopWords(value: Boolean): this.type = set(shouldRemoveStopWords, value)
  setDefault(shouldRemoveStopWords -> true)

  override protected def createTransformFunc: String => Seq[String] = { originStr =>
    HanLP.Config.ShowTermNature = false
    HanLP.Config.Normalization = false
    val segment = HanLP.newSegment()
    val termList = segment.seg(HanLP.convertToSimplifiedChinese(originStr.toLowerCase))

    if ($(shouldRemoveStopWords)) {
      CoreStopWordDictionary.apply(termList)
    }

    termList.toArray.map(_.asInstanceOf[Term].word)
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