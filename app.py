import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

st.set_page_config(
    page_title="RAG AI Assistant",
    page_icon="🤖",
    layout="wide"
)

st.markdown("""
    <style>
    .main { padding: 1.5rem; }
    .stChatMessage { margin-bottom: 0.8rem; }
    .stButton>button { width: 100%; border-radius: 8px; }
    </style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.title("📚 Document Portal")
    st.caption("Upload your PDF to start chatting with AI")

    uploaded_file = st.file_uploader("Upload PDF File", type=["pdf"])

    st.divider()
    st.subheader("⚙️ Settings")
    selected_model = st.selectbox(
        "Select Model",
        ["llama-3.1-8b-instant", "llama3-8b-8192", "mixtral-8x7b-32768"],
        index=0
    )

    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

st.title("🤖 RAG Knowledge Assistant")
st.write("Upload a document from the sidebar and ask any question below.")

if "messages" not in st.session_state:
    st.session_state.messages = []


@st.cache_resource(show_spinner="Processing and indexing document...")
def process_pdf(_file_bytes, file_name):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(_file_bytes)
        tmp_path = tmp_file.name

    try:
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
        splits = text_splitter.split_documents(docs)

        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
        return vectorstore
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


if uploaded_file:
    vectorstore = process_pdf(uploaded_file.getvalue(), uploaded_file.name)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    try:
        llm = ChatGroq(model=selected_model, temperature=0.1)
    except Exception:
        llm = ChatGroq(model_name=selected_model, temperature=0.1)

    system_prompt = (
        "You are an expert academic and research assistant. "
        "Use the following pieces of retrieved context to answer the user's question clearly and concisely. "
        "If the answer cannot be found in the context, state politely that you do not know based on the provided document.\n\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_query := st.chat_input("Ask something about the uploaded document..."):
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing document..."):
                try:
                    response = rag_chain.invoke({"input": user_query})
                    answer = response["answer"]
                    st.markdown(answer)

                    st.session_state.messages.append({"role": "assistant", "content": answer})

                    with st.expander("📌 View Source Excerpts"):
                        for idx, doc in enumerate(response["context"]):
                            st.markdown(f"**Source Chunk {idx + 1}:**")
                            st.caption(doc.page_content)
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")
else:
    st.info("👈 Please upload a PDF document from the sidebar to begin.")