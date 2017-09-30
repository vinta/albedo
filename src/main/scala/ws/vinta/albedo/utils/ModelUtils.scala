package ws.vinta.albedo.utils

import org.apache.hadoop.mapred.InvalidInputException
import org.apache.spark.ml.util.{MLReadable, MLWritable}

object ModelUtils {
  def loadOrCreateModel[T <: MLWritable](ModelClass: MLReadable[T], path: String, createModelFunc: () => T): T = {
    try {
      ModelClass.load(path)
    } catch {
      case e: InvalidInputException => {
        if (e.getMessage.contains("Input path does not exist")) {
          val model = createModelFunc()
          model.write.overwrite().save(path)
          model
        } else {
          throw e
        }
      }
    }
  }
}