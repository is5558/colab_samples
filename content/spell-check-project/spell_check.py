import sparknlp
from pyspark.sql import SparkSession
from sparknlp.base import DocumentAssembler
from sparknlp.annotator import *
from pyspark.ml import Pipeline

def initialize_spark_nlp():
    spark = SparkSession.builder \
        .appName("check_spelling") \
        .config("spark.jars", "spark-nlp-5.1.3.jar") \
        .getOrCreate()
    return spark

def main():
    spark = initialize_spark_nlp()
    print("Spark NLP version:", sparknlp.version())
    print("*" * 77)

    data = spark.createDataFrame([["Yesturday, I went to the libary to borow a book about anciant civilizations. "
        "The wether was pleasent, so I decidid to walk insted of taking the buss. On the way, "
        "I saw a restuarent that lookt intresting, and I plan to viset it soon."]]).toDF("text")

    document = DocumentAssembler().setInputCol("text").setOutputCol("document")
    tokenizer = Tokenizer().setInputCols(["document"]).setOutputCol("token")
    spell = NorvigSweetingModel.pretrained() \
        .setInputCols(["token"]) \
        .setOutputCol("spell")

    pipeline = Pipeline(stages=[document, tokenizer, spell])
    model = pipeline.fit(data)
    result = model.transform(data)

    result.select("spell.result").show(truncate=False)

if __name__ == "__main__":
    main()
