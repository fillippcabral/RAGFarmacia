import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains.retrieval_qa.base import RetrievalQA

# === 1. Configuração inicial ===
load_dotenv()
st.set_page_config(page_title="🔬 Consulta de medicamentos", layout="wide")

pdf_file = st.file_uploader("📄 Envie o documento do medicamento com os lotes com o Manual de uso (PDF ou CSV)", type=["pdf", "csv"])

if pdf_file:
    # Cria diretório temporário para salvar o PDF
    os.makedirs("docs", exist_ok=True)
    pdf_path = os.path.join("docs", pdf_file.name)

    with open(pdf_path, "wb") as f:
        f.write(pdf_file.read())

    st.info("📘 Extraindo conteúdo do PDF...")

    try:
        loader = PyMuPDFLoader(pdf_path)
        documents = loader.load()
    except Exception as e:
        st.error(f"Erro ao ler o PDF: {e}")
        st.stop()

    # === 4. Fragmentação do texto ===
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = splitter.split_documents(documents)

    # === 5. Criação dos embeddings ===
    embeddings = OpenAIEmbeddings(openai_api_key=openai_key)

    # === 6. Banco vetorial (FAISS) ===
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    # === 7. Configuração do modelo de linguagem ===
    llm = ChatOpenAI(
        api_key=openai_key,
        model="gpt-4o-mini",  # Modelo moderno, rápido e econômico
        temperature=0.2
    )

    # === 8. Criação da cadeia RAG ===
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True
    )

    st.success("✅ Medicamento processado e indexado com sucesso!")

    # === 9. Interface de Perguntas ===
    st.subheader("🩺 Pergunte sobre o medicamento")
    user_question = st.text_input("❓ Digite sua pergunta:")

    if user_question:
        with st.spinner("🔎 Analisando os resultados com IA..."):
            resposta = rag_chain.invoke({"query": user_question})

        st.markdown("### 🧠 Resposta da IA:")
        st.markdown(resposta["result"])

        # Exibe as fontes de referência do texto
        with st.expander("📚 Fontes de contexto utilizadas"):
            for i, doc in enumerate(resposta["source_documents"], start=1):
                st.markdown(f"**Trecho {i}:** {doc.page_content[:300]}...")
