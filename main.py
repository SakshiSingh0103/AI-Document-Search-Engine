from indexer import build_index
from search import search_documents

documents, filenames, vectorizer, tfidf_matrix = build_index()

query = input("Enter your search query: ").lower()

results = search_documents(query, documents, filenames, vectorizer, tfidf_matrix)

print("\nSearch Results:")

for filename, document, score in results:
    if score > 0:
        print(f"\n{filename} : {score:.4f}")
        print(document)