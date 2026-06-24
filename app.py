import streamlit as st

st.set_page_config(
    page_title="AI Document Search Engine",
    page_icon="🔍",
    layout="wide"
)

st.sidebar.title("📊 AI Search Engine")

st.sidebar.info("""
Search documents using:

- TF-IDF
- Cosine Similarity
- Streamlit UI
""")

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
    st.info(f"Found {len([r for r in results if r[2] > 0])} matching documents")

    for filename, document, score in results:
        if score > 0:
             st.success(f"📄 {filename}")
             st.caption("Matching document")
             st.progress(float(score))
             st.metric("Similarity Score", f"{score*100:.2f}%")
             st.write(document)
             st.divider()
             st.markdown("---")
st.markdown("Developed by Sakshi Singh 🚀")
