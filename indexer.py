import os
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer

def build_index():
    documents = []
    filenames = []

    # Read TXT files
    if os.path.exists("documents"):
        for file in os.listdir("documents"):
            if file.endswith(".txt"):
                with open(
                    os.path.join("documents", file),
                    "r",
                    encoding="utf-8"
                ) as f:
                    documents.append(f.read())
                    filenames.append(file)

    # Read PDF files
    if os.path.exists("pdfs"):
        for file in os.listdir("pdfs"):
            if file.endswith(".pdf"):
                pdf_text = ""

                reader = PdfReader(
                    os.path.join("pdfs", file)
                )

                for page in reader.pages:
                    pdf_text += page.extract_text() or ""

                documents.append(pdf_text)
                filenames.append(file)

    vectorizer = TfidfVectorizer(
        stop_words="english"
    )

    tfidf_matrix = vectorizer.fit_transform(documents)

    return (
        documents,
        filenames,
        vectorizer,
        tfidf_matrix
    )