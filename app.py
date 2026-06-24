import streamlit as st

st.set_page_config(
    page_title="AI Document Search Engine",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 AI Document Search Engine")
st.caption("Search documents using TF-IDF and Cosine Similarity")

from indexer import build_index
from search import search_documents

documents, filenames, vectorizer, tfidf_matrix = build_index()

query = st.text_input("🔎 Search Documents")

if query:
    results = search_documents(
        query.lower(),
        documents,
        filenames,
        vectorizer,
        tfidf_matrix
    )

    st.subheader("Search Results")

    for filename, document, score in results:
        if score > 0:
            st.write(f"📄 {filename} (Score: {score:.4f})")
            st.write(document)
            st.write("---")