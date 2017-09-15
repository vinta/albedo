package ws.vinta.albedo.utils

import scala.util.matching.Regex

object StringUtils {
  val wordPatternEngOnly = """\w\.\-_"""
  val wordPatternIncludeCJK = """\w\.\-_\p{InHiragana}\p{InKatakana}\p{InBopomofo}\p{InCJKCompatibilityIdeographs}\p{InCJKUnifiedIdeographs}"""

  val reExtractWords: Regex = s"([$wordPatternEngOnly]+)".r
  val reExtractWordsIncludeCJK: Regex = s"([$wordPatternIncludeCJK]+)".r
  val reExtractEmailDomain: Regex = s"@([$wordPatternEngOnly]+)".r

  def extractWords(text: String): List[String] = {
    reExtractWords.findAllIn(text).toList
  }

  def extractWordsIncludeCJK(text: String): List[String] = {
    reExtractWordsIncludeCJK.findAllIn(text).toList
  }

  def extractEmailDomain(email: String): String = {
    try {
      reExtractEmailDomain.findAllIn(email).matchData.toList(0).group(1)
    } catch {
      case _: IndexOutOfBoundsException => {
        email
      }
    }
  }
}