import os
from sklearn.feature_extraction.text import TfidfVectorizer

def build_index():
    documents = []
    filenames = []

    for file in os.listdir("documents"):
        with open("documents/" + file, "r", encoding="utf-8") as f:
            documents.append(f.read())
            filenames.append(file)

    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(documents)

    return documents, filenames, vectorizer, tfidf_matrix