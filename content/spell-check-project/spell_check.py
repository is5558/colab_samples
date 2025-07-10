import sparknlp
from sparknlp.pretrained import PretrainedPipeline

def initialize_spark_nlp():
    spark = sparknlp.start()
    print("Spark NLP version:", sparknlp.version())
    return spark

def load_pipeline(pipeline_name='check_spelling', lang='en'):
    return PretrainedPipeline(pipeline_name, lang=lang)

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
    spark = initialize_spark_nlp()
    pipeline = load_pipeline()
    annotations = pipeline.fullAnnotate(text)[0]
    corrected_text = get_corrected_text(annotations)

    print("*" * 77)
    print("Original Text:\n", text)
    print("Corrected Text:\n", corrected_text)
    print("*" * 77)

if __name__ == "__main__":
    main()
