import streamlit as st

from indexer import build_index
from search import search_documents

st.set_page_config(page_title="AI Search Engine", page_icon="🔍")

st.title("🔍 AI Document Search Engine")
st.write("Search through your documents using Natural Language Processing")

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