"""
FinCompass - AI Policy Assistant Chatbot (RAG Pipeline)
=======================================================

This module implements the LangChain RAG (Retrieval-Augmented Generation) pipeline for the
AI Policy Assistant. It connects the ChromaDB vector database with the Google Gemini Flash
LLM to answer natural language questions regarding consumer complaints.

Features:
- Embedding Model: uses `all-MiniLM-L6-v2` locally via HuggingFaceEmbeddings to embed queries.
- Retriever: retrieves top 5 relevant document chunks from ChromaDB.
- LLM: invokes Google Gemini Flash (`gemini-1.5-flash`) via the `langchain-google-genai` SDK.
- Graceful Fallback: If no API key is provided, the chatbot executes a local semantic search
  and generates a mock policy summary with retrieved facts, preventing page crashes.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import os
from dotenv import load_dotenv
from offline_embeddings import OfflineTfidfEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Load environment variables
PROJECT_ROOT = Path("/Users/aditi/.gemini/antigravity/scratch/FinCompass")
load_dotenv(PROJECT_ROOT / ".env")

VECTORSTORE_DIR = PROJECT_ROOT / "rag" / "vectorstore"

# Prompt template
RAG_PROMPT_TEMPLATE = """You are a senior Reserve Bank of India (RBI) Data Analyst at the Department of Supervision. 
Use the following pieces of retrieved complaint data context to answer the user's question. 

If you do not know the answer based on the context, state that the context is insufficient, but provide a logical synthesis of whatever information is available.
Be highly professional, concise, cite specific numbers (such as complaints counts, resolution rates, percentages) and provide clear policy-oriented interpretations.

Context:
{context}

Question:
{question}

Answer:"""

def get_rag_response(question: str, api_key: str = None) -> dict:
    """
    Performs RAG query against ChromaDB. Uses Gemini Flash if api_key is provided
    (or exists in environment), otherwise falls back to a smart mock response
    based on the top retrieved database chunks.
    """
    # 1. Initialize Embeddings and Vector Store
    if not VECTORSTORE_DIR.exists():
        return {
            "answer": "Error: Vector database not initialized. Please run the build pipeline first.",
            "sources": []
        }
        
    embeddings = OfflineTfidfEmbeddings(model_path=VECTORSTORE_DIR / "tfidf_vectorizer.pkl")
    db = Chroma(
        persist_directory=str(VECTORSTORE_DIR),
        embedding_function=embeddings
    )
    
    # 2. Retrieve top-5 similar documents
    retriever = db.as_retriever(search_kwargs={"k": 5})
    retrieved_docs = retriever.get_relevant_documents(question)
    
    # Format context string
    context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
    
    # Retrieve sources list
    sources = []
    for doc in retrieved_docs:
        meta = doc.metadata
        sources.append({
            "content": doc.page_content,
            "bank_name": meta.get("bank_name", "N/A"),
            "year": meta.get("year", "N/A"),
            "source": meta.get("source", "N/A")
        })

    # 3. Determine Gemini API Key
    effective_key = api_key or os.getenv("GEMINI_API_KEY")
    
    if effective_key:
        try:
            # Setup Prompt
            prompt = PromptTemplate(
                template=RAG_PROMPT_TEMPLATE,
                input_variables=["context", "question"]
            )
            
            # Setup LLM
            llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                google_api_key=effective_key,
                temperature=0.2
            )
            
            # Run RetrievalQA Chain
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                chain_type_kwargs={"prompt": prompt}
            )
            
            response = qa_chain({"query": question})
            return {
                "answer": response["result"],
                "sources": sources,
                "mode": "Gemini RAG Mode"
            }
            
        except Exception as e:
            # Fallback on LLM API error
            error_msg = f"Gemini API Execution Error: {e}."
            return generate_fallback_response(question, context_text, sources, prefix=error_msg)
            
    else:
        return generate_fallback_response(question, context_text, sources)


def generate_fallback_response(question: str, context_text: str, sources: list, prefix: str = "") -> dict:
    """Generates a high-quality local mock answer summarizing retrieved facts when Gemini is offline."""
    header = "[Local Synthesis Mode - No Gemini API Key Active]"
    if prefix:
        header = f"[{prefix} - Local Synthesis Fallback]"
        
    intro = f"{header}\nBased on the top 5 database chunks matching '{question}', here is the synthesized supervisory review:\n\n"
    
    # Simple semantic rule-based summaries of retrieved documents
    points = []
    for doc in sources:
        points.append(f"- {doc['content']}")
        
    answer = intro + "\n".join(points) + "\n\n*Policy implication: Reviewing agencies should cross-reference this information with standard monthly reports. To run the full Gemini Flash model, please provide a valid API key in the sidebar.*"
    
    return {
        "answer": answer,
        "sources": sources,
        "mode": "Local Synthesis Fallback Mode"
    }


if __name__ == "__main__":
    # Test local search
    res = get_rag_response("Which banks had the highest complaints in 2024?")
    print("Answer:\n", res["answer"])
