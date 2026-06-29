"""
FinCompass - Vector Database Builder (RAG Pipeline)
===================================================

This module builds the vector database (ChromaDB) for the RAG policy assistant. It
translates structured records from SQLite tables (`monthly_summary`, `policy_flags`)
and analytical trends into rich, natural language narratives. These narratives are
then embedded and stored in a vector index to enable contextual semantic search
for Gemini Flash.

Steps:
1. Fetch structured aggregates from `monthly_summary` and `policy_flags` in SQLite.
2. Narrative Generation: Convert rows into descriptive sentences (e.g. 'In December 2024,
   SBI had 250 complaints, with a resolution rate of 88%...').
3. Vector Ingestion: Embed text chunks using the free local `all-MiniLM-L6-v2` embedding
   model (via HuggingFaceEmbeddings) and persist to `rag/vectorstore/`.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import sqlite3
import pandas as pd
from datetime import datetime
from langchain.docstore.document import Document
from offline_embeddings import OfflineTfidfEmbeddings
from langchain_community.vectorstores import Chroma

PROJECT_ROOT = Path("/Users/aditi/.gemini/antigravity/scratch/FinCompass")
DB_PATH = PROJECT_ROOT / "database" / "fincompass.db"
VECTORSTORE_DIR = PROJECT_ROOT / "rag" / "vectorstore"

def get_monthly_narratives(conn: sqlite3.Connection) -> list[Document]:
    """Retrieves monthly summaries and converts them to narrative documents."""
    query = """
        SELECT 
            b.bank_name,
            b.bank_type,
            ms.year,
            ms.month,
            ms.total_complaints,
            ms.resolved_count,
            ms.pending_count,
            ms.escalated_count,
            ms.avg_resolution_days,
            ms.complaint_growth_pct
        FROM monthly_summary ms
        JOIN banks b ON ms.bank_id = b.bank_id
    """
    df = pd.read_sql_query(query, conn)
    docs = []
    
    for _, row in df.iterrows():
        month_name = datetime(int(row["year"]), int(row["month"]), 1).strftime("%B")
        res_rate = round((row["resolved_count"] / row["total_complaints"]) * 100, 1) if row["total_complaints"] > 0 else 0.0
        
        text = (
            f"In {month_name} {int(row['year'])}, {row['bank_name']} ({row['bank_type']}) received a total of "
            f"{int(row['total_complaints'])} consumer complaints. Out of these, {int(row['resolved_count'])} were "
            f"resolved (resolution rate of {res_rate}%), {int(row['pending_count'])} remain pending, and "
            f"{int(row['escalated_count'])} were escalated. "
            f"The average resolution time for resolved complaints was {row['avg_resolution_days']} days. "
            f"The month-over-month complaint volume growth for this bank was {row['complaint_growth_pct']}%."
        )
        
        metadata = {
            "source": "monthly_summary",
            "bank_name": row["bank_name"],
            "year": int(row["year"]),
            "month": int(row["month"]),
            "type": "summary"
        }
        
        docs.append(Document(page_content=text, metadata=metadata))
        
    return docs

def get_policy_flag_narratives(conn: sqlite3.Connection) -> list[Document]:
    """Retrieves policy/supervisory alerts and converts them to documents."""
    query = """
        SELECT 
            b.bank_name,
            pf.year,
            pf.quarter,
            pf.flag_type,
            pf.flag_description,
            pf.severity
        FROM policy_flags pf
        JOIN banks b ON pf.bank_id = b.bank_id
    """
    df = pd.read_sql_query(query, conn)
    docs = []
    
    for _, row in df.iterrows():
        text = (
            f"Regulatory Alert: In {int(row['year'])} Quarter {int(row['quarter'])}, {row['bank_name']} was flagged for "
            f"'{row['flag_type']}' with {row['severity']} severity. Description: {row['flag_description']}"
        )
        
        metadata = {
            "source": "policy_flags",
            "bank_name": row["bank_name"],
            "year": int(row["year"]),
            "quarter": int(row["quarter"]),
            "type": "flag"
        }
        
        docs.append(Document(page_content=text, metadata=metadata))
        
    return docs

def build_vector_database():
    """Builds and stores Chroma vector store."""
    print("Building narratives from database tables...")
    
    # 1. Fetch narratives
    with sqlite3.connect(str(DB_PATH)) as conn:
        docs = []
        docs.extend(get_monthly_narratives(conn))
        docs.extend(get_policy_flag_narratives(conn))
        
    print(f"Generated {len(docs)} document chunks.")
    
    # Clean vectorstore directory if exists to ensure fresh build
    if VECTORSTORE_DIR.exists():
        import shutil
        shutil.rmtree(VECTORSTORE_DIR)
        
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    
    # 2. Ingest into Chroma using local TF-IDF embeddings
    print("Initializing local OfflineTfidfEmbeddings...")
    embeddings = OfflineTfidfEmbeddings(model_path=VECTORSTORE_DIR / "tfidf_vectorizer.pkl")
    
    print("Storing narratives in persistent ChromaDB index...")
    db = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=str(VECTORSTORE_DIR)
    )
    db.persist()
    print(f"Successfully created vector database at: {VECTORSTORE_DIR}")


if __name__ == "__main__":
    build_vector_database()
