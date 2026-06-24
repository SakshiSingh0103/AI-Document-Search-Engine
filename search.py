from sklearn.metrics.pairwise import cosine_similarity

def search_documents(query, documents, filenames, vectorizer, tfidf_matrix):
    query_vector = vectorizer.transform([query])

    similarity_scores = cosine_similarity(query_vector, tfidf_matrix)[0]

    results = list(zip(filenames, documents, similarity_scores))

    results.sort(key=lambda x: x[2], reverse=True)

    return results