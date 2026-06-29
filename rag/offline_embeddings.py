"""
FinCompass - Local Offline Text Embeddings
==========================================

This module implements a custom, local-only text embedding class for LangChain.
By using scikit-learn's `TfidfVectorizer`, it generates high-quality text vectors
without making any outbound HTTP requests, solving lockups and proxy constraints in
network-restricted environments.

The vectorizer state is persisted as a pickle file inside the vector store directory.
"""

import pickle
from pathlib import Path
from langchain_core.embeddings import Embeddings
from sklearn.feature_extraction.text import TfidfVectorizer

class OfflineTfidfEmbeddings(Embeddings):
    """
    Local, internet-free text embedding implementation using scikit-learn TF-IDF.
    Ensures zero external API dependency and lightning-fast execution times.
    """
    def __init__(self, model_path: Path):
        self.model_path = Path(model_path)
        self.vectorizer = TfidfVectorizer(max_features=128, stop_words="english")
        self.is_fitted = False
        
        # Load from disk if vectorizer already exists
        if self.model_path.exists():
            try:
                with open(self.model_path, "rb") as f:
                    self.vectorizer = pickle.load(f)
                self.is_fitted = True
            except Exception:
                self.is_fitted = False

    def fit(self, texts: list[str]):
        """Fits the TF-IDF vectorizer on the corpus documents and saves to disk."""
        self.vectorizer.fit(texts)
        self.is_fitted = True
        
        # Save vectorizer state
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.model_path, "wb") as f:
            pickle.dump(self.vectorizer, f)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Computes embeddings for a list of document strings."""
        if not self.is_fitted:
            self.fit(texts)
            
        vectors = self.vectorizer.transform(texts).toarray()
        return [v.tolist() for v in vectors]

    def embed_query(self, text: str) -> list[float]:
        """Computes embedding for a single search query."""
        if not self.is_fitted:
            # Fallback if querying before any fit occurred
            self.fit([text])
            
        vector = self.vectorizer.transform([text]).toarray()
        return vector[0].tolist()
