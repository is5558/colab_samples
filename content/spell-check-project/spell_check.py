import sparknlp
from pyspark.sql import SparkSession
from sparknlp.base import DocumentAssembler
from sparknlp.annotator import *
from pyspark.ml import Pipeline

def initialize_spark_nlp():
    spark = SparkSession.builder \
        .appName("check_spelling") \
        .config("spark.jars", "/content/spark-nlp-6.0.5.jar") \
        .getOrCreate()
    return spark

def load_pipeline(pipeline_name='check_spelling', lang='en'):
    try:
        return PretrainedPipeline(pipeline_name, lang=lang)
    except Exception as e:
        print(f"Error loading pipeline '{pipeline_name}':", str(e))
        raise

def get_corrected_text(annotations):
    try:
        corrected_tokens = [token.result for token in annotations['checked']]
        return " ".join(corrected_tokens).replace(" ,", ",").replace(" .", ".")
    except KeyError:
        print("Error: 'checked' key not found in annotations.")
        return ""

def main():
    text = (
        "Yesturday, I went to the libary to borow a book about anciant civilizations. "
        "The wether was pleasent, so I decidid to walk insted of taking the buss. On the way, "
        "I saw a restuarent that lookt intresting, and I plan to viset it soon."
    )

    try:
        # Initialize Spark NLP and load the pipeline
        spark = initialize_spark_nlp()
        pipeline = load_pipeline()

        # Annotate text
        annotations = pipeline.fullAnnotate(text)[0]

        # Get and print corrected text
        corrected_text = get_corrected_text(annotations)
        print("*"*77)
        print("Original Text:\n", text)
        print("Corrected Text:\n", corrected_text)
        print("*"*77)

    except Exception as e:
        print("An unexpected error occurred:", str(e))

if __name__ == "__main__":
    main()
