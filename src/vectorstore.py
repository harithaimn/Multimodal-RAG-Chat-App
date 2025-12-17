# src/vectorstore.py

import os
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings

load_dotenv()


class VectorDB:
    """
    Minimal Pinecone connector for Multimodal RAG.

    Responsibilities:
    - Embed query text
    - Query Pinecone with optional metadata filters
    - Return formatted results for RAGChain
    """

    def __init__(self, embedding_model: str = None):
        # --- Env ---
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        self.index_name = os.getenv("PINECONE_INDEX_NAME")

        if not self.pinecone_api_key or not self.index_name:
            raise ValueError("PINECONE_API_KEY and PINECONE_INDEX_NAME must be set")

        # --- Pinecone client ---
        try:
            self.pc = Pinecone(api_key=self.pinecone_api_key)
        except Exception as e:
            raise ConnectionError(f"Pinecone init failed: {e}")

        # --- Verify index ---
        existing_indexes = self.pc.list_indexes().names()
        if self.index_name not in existing_indexes:
            raise ValueError(
                f"Pinecone index '{self.index_name}' not found. Run ingest.py first."
            )

        self.index = self.pc.Index(self.index_name)

        # --- Embeddings ---
        model_name = embedding_model or os.getenv(
            "EMBEDDING_MODEL", "text-embedding-3-small"
        )
        self.embeddings = OpenAIEmbeddings(model=model_name)

        print(f"✅ Connected to Pinecone index '{self.index_name}'")

    # --------------------------------------------------
    # Retriever (callable)
    # --------------------------------------------------
    def as_retriever(self, search_kwargs: dict = None):
        """
        Returns a simple callable retriever.

        retriever(query, filters, k) -> List[dict]
        """
        if search_kwargs is None:
            search_kwargs = {"k": 5}

        def retriever(
            query: str,
            filters: dict = None,
            k: int = None,
            min_score: float = 0.0,
        ):
            top_k = k or search_kwargs.get("k", 5)

            # --- Build Pinecone metadata filter ---
            pinecone_filter = {}
            if filters:
                for key, value in filters.items():
                    if value and value != "All":
                        pinecone_filter[key] = {"$eq": value}

            # --- Embed query ---
            query_vec = self.embeddings.embed_query(query)

            # --- Query Pinecone ---
            results = self.index.query(
                vector=query_vec,
                top_k=top_k,
                include_metadata=True,
                filter=pinecone_filter or None,
            )

            matches = results.get("matches", [])

            # --- Format output ---
            formatted = []
            for m in matches:
                if m["score"] < min_score:
                    continue

                meta = m.get("metadata", {}) or {}

                formatted.append(
                    {
                        "id": m.get("id"),
                        "score": m.get("score"),
                        "metadata": meta,
                        "image_url": meta.get("image_url"),
                        "caption": meta.get("caption"),
                    }
                )

            return formatted

        return retriever

    # --------------------------------------------------
    # Direct query helper
    # --------------------------------------------------
    def query(self, query: str, filters: dict = None, k: int = 5):
        retriever = self.as_retriever({"k": k})
        return retriever(query, filters=filters, k=k)
