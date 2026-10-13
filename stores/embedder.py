import numpy as np
from langchain_core.embeddings.embeddings import Embeddings


class Embedder:
    def __init__(self, embedder: Embeddings):
        """Wrapper to normalize embeddings produced by a LangChain embedder.

        Args:
            embedder: Underlying embeddings implementation (e.g., AzureOpenAIEmbeddings).
        """
        self.embedder = embedder

    def embed_items(self, texts: str | list[str]) -> np.ndarray:
        """Embed one or many texts and return L2-normalized vectors."""
        if isinstance(texts, str):
            texts = [texts]

        embeddings = self.embedder.embed_documents(texts)
        embeddings_array = np.array(embeddings, dtype=np.float32)
        normalized_embeddings = self._l2norm(embeddings_array)

        return normalized_embeddings

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query string and return an L2-normalized vector."""
        embedding = self.embedder.embed_query(text)
        embedding_array = np.array(embedding, dtype=np.float32)
        normalized_embedding = self._l2norm(embedding_array)

        return normalized_embedding

    @staticmethod
    def _l2norm(x: np.ndarray) -> np.ndarray:
        """L2-normalize vectors along the last axis with numerical stability."""
        return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)
