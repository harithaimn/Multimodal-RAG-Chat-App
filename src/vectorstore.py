import os
from dotenv import load_dotenv
from pinecone import Pinecone, PineconeException
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from src.context_rules import  build_metadata_filters

load_dotenv()

class VectorDB:
    """
    A connector class for a pre-populated Pinecone vector database.
    Supports filtered RAG retrieval (e.g, by industry, campaign, or keyword)
    Its sole purpose is to connect to the index and provide a retriever.
    -Filtered search via dropdown filters
    -Image-aware retrieval (image_url + caption stored during ingest)
    """
    def __init__(self, embedding_model: str = "text-embedding-3-small"):
        """
        Initializes the connection to the Pinecone index.
        """
        # -----------------------------------------------
        # Load Environment Variables
        # -----------------------------------------------
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        self.index_name = os.getenv("PINECONE_INDEX_NAME")
        
        if not self.pinecone_api_key:
            raise ValueError("❌ PINECONE_API_KEY is not set in the .env file.")
        
        if not self.index_name:
            raise ValueError("❌ PINECONE_INDEX_NAME is not set in the .env file.")
        
        # ----------------------------------------------
        # Initialize Pinecone client
        try:
            self.pc = Pinecone(api_key=self.pinecone_api_key)
        except PineconeException as e:
            raise ConnectionError(f"❌ Pinecone initialization failed: {str(e)}")
        
        # Verify that the index exists
        existing_indexes = self.pc.list_indexes().names()
        if self.index_name not in existing_indexes:
            raise ValueError(f"❌ Pinecone index '{self.index_name}' not found. Please run ingest.py first.")

        # Initialize the embedding model
        self.embedding_model = OpenAIEmbeddings(model=embedding_model)

        # Initialize the LangChain PineconeVectorStore wrapper
        self.vectorstore = PineconeVectorStore(index_name=self.index_name, embedding=self.embedding_model)
        
        print(f"✅ Connected to Pinecone index: '{self.index_name}' using {embedding_model}")


    # ---------------------------------------------------
    # Retrieve Documents
    # ---------------------------------------------------
    def as_retriever(self, search_kwargs: dict={'k': 5}, filters: dict = None):
        """
        Returns the vector store instance configured as a retriever.
        
        Args:
            search_kwargs (dict): A dictionary to configure search parameters,
                                  such as the number of documents to retrieve ('k').
        
        Returns:
            A LangChain retriever object.
        """
        pinecone_filters = build_metadata_filters(filters)

        return self.vectorstore.as_retriever(
            search_kwargs={**search_kwargs, "filters": pinecone_filters}
            )
    
    # --------------------------------------------------------
    # Direct Query Interface
    # --------------------------------------------------------
    #def query(self, query_text: str, top_k: int=5, filters: dict=None):
    def query(self, query_text: str, top_k: int = 5, filters: dict = None, min_score: float = 0.0):

        """
        Query Pinecone directly for raw vector matches (bypasses LangChain retriever).

        Args: 
            query_text (str): The text to embed and query.
            top_k (int): Number of results to return.
            fitler (dict): Optional metadata filter (e.g., {"campaign_name": "Bryan"}).

        Returns:
            List of matched documents with metadata.
        """
        # Compute query embedding
        try:
            query_embedding = self.embedding_model.embed_query(query_text)
        except Exception as e:
            raise RuntimeError(f"Failed to embed query: {e}")
        
        index = self.pc.Index(self.index_name)

        # Build Pinecone metadata filter automatically using DROPDOWN_CONTEXTS
        pinecone_filter = build_metadata_filters(filters)
        
        # if filters:
        #     for key, value in filters.items():
        #         if value and value != "All":
        #             pinecone_filter[key] = {"$eq": value}
        
        # Perform query
        try:
            results = index.query(
                vector = query_embedding,
                top_k = top_k,
                include_metadata=True,
                filter=pinecone_filter
            )
        except Exception as e:
            print(f"⚠️ Pinecone query failed: {e}")
            return []
        
        matches = results.get("matches", [])

        # Format results
        formatted = []
        for m in matches:
            if m["score"] < min_score:
                continue

            meta = m.get("metadata", {})
            
            formatted.append({
                "id": m["id"],
                "score": m["score"],
                "metadata": meta.get("text", ""),
                "metadata": meta,
                "image_url": meta.get("image_url"),
                "caption": meta.get("caption")
            })
        
        print(f"📌 Retrieved {len(formatted)} documents (filters={filters})")

        # formatted = [
        #     {
        #         "id": match["id"],
        #         "score": match["score"],
        #         "metadata": match["metadata"]
        #     }
        #     for match in results.get("matches", [])
        #     if match["score"] >= min_score
        # ]

        if not formatted:
            print(f" No matches found for query: '{query_text}' with filters {filters}")
        else:
            print(f" Retrieved {len(formatted)} matches for '{query_text}' with filters {filters}")
        
        return formatted

    # =====================================================
    # 3. Category-Aware Search Helper
    # =====================================================
    def query_by_category(self, query_text: str, category: str, top_k: int = 5, min_score: float = 0.0):

        """
        Specialized method to search within a specific dataset category.

        Example:
            vectordb.query_by_category("new burger ad ideas", category="burger f&b")
        """
        #filter = {"category": {"$eq": category}}
        #return self.query(query_text=query_text, top_k=top_k, filters=filter)
        return self.query(query_text, top_k=top_k, filters={"category": category}, min_score=min_score)
    
    # --------------------------------
    # Inspect Stored Data
    # --------------------------------
    #def list_all(self, limit: int = 10):
    def list_all(self, verbose: bool = False):
        """
        Lists index statistics (for debugging).

        Args:
            verbose (bool): Wether to print detailed metadata counts.

        Returns:
            dict: Index statistics from Pinecone.
        """
        try:
            index = self.pc.Index(self.index_name)
            stats = index.describe_index_stats()
            
            total = stats.get("total_vector_count", 0)
            print(f"Total vectors in '{self.index_name}': {total}")
            #total = stats["total_vector_count"]
            #print(f" Total vectors in '{self.index_name}": {total})
            #print(f"Total vectors in '{self.index_name}': {stats['total_vector_count']}")
            
            if verbose:
                print("📁 Full index stats:")
                print(stats)
            #return stats

            # Note: Pinecone doesn't provide direct vector listing
            # So this only prints metadatacounts.
            return stats
        
        except Exception as e:
            print(f"⚠️ Failed to get index stats: {e}")
            return {}

