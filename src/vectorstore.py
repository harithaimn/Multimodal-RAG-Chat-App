import os
from dotenv import load_dotenv
from pinecone import Pinecone, PineconeException
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from src.context_rules import DROPDOWN_CONTEXTS

load_dotenv()

class VectorDB:
    """
    A connector class for a pre-populated Pinecone vector database.
    Supports filtered RAG retrieval (e.g, by industry, campaign, or keyword)
    Its sole purpose is to connect to the index and provide a retriever.
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
    def as_retriever(self, search_kwargs: dict={'k': 5}):
        """
        Returns the vector store instance configured as a retriever.
        
        Args:
            search_kwargs (dict): A dictionary to configure search parameters,
                                  such as the number of documents to retrieve ('k').
        
        Returns:
            A LangChain retriever object.
        """
        return self.vectorstore.as_retriever(search_kwargs=search_kwargs)
    
    # --------------------------------------------------------
    # Direct Query Interface
    # --------------------------------------------------------
    def query(self, query_text: str, top_k: int=5, filters: dict=None):

        """
        Query Pinecone directly for raw vector matches (bypasses LangChain retriever).

        Args: 
            query_text (str): The text to embed and query.
            top_k (int): Number of results to return.
            fitler (dict): Optional metadata filter (e.g., {"campaign_name": "Bryan"}).

        Returns:
            List of matched documents with metadata.
        """
        if filters is None:
            filters = {}

        # Build Pinecone-compatible metadata filter
        metadata_filter = {}

        # Apply industry filter
        if filters.get("industry") and filters["industry"] != "All":
            metadata_filter["industry"] = {"$eq": filters["industry"]}

        # Apply campaign objective filter
        if filters.get("campaign_objective") and filters["campaign_objective"] != "All":
            metadata_filter["campaign_objective"] = {"$eq": filters["campaign_objective"]}

        # Apply ads language filter
        if filters.get("ads_language") and filters["ads_language"] != "All":
            metadata_filter["ads_language"] = {"$eq": filters["ads_language"]}
        
        # Optionally add any other dropdown filters from context_rules.py
        for key in filters:
            if key not in ["industry", "campaign_objective", "ads_language", "target_market"]:
                val = filters[key]
                if val and val != "All":
                    metadata_filter[key] = {"$eq": val}
        
        try:
            query_embedding = self.embedding_model.embede_query(query_text)
            index = self.pc.Index(self.index_name)

            # Perform the query
            results = index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                filter=metadata_filter
            )

            formatted = [
                {"id": match["id"], "score": match["score"], "metadata": match["metadata"]}
                for match in results.get("matches", [])
            ]

            if not formatted:
                print(f"⚠️ No matches found for query: '{query_text}' with filters {metadata_filter}")
            else:
                print(f"🔍 Retrieved {len(formatted)} matches for '{query_text}' with filters {metadata_filter}")

            return formatted
        except Exception as e:
            print(f"❌ Query failed: {e}")
            return []
    
    # =====================================================
    # 3. Category-Aware Search Helper
    # =====================================================
    def query_by_category(self, query_text: str, category: str, top_k: int = 5):
        """
        Specialized method to search within a specific dataset category.

        Example:
            vectordb.query_by_category("new burger ad ideas", category="burger f&b")
        """
        filter = {"category": {"$eq": category}}
        return self.query(query_text=query_text, top_k=top_k, filters=filter)
    
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
            
            #total = stats["total_vector_count"]
            #print(f" Total vectors in '{self.index_name}": {total})
            print(f"Total vectors in '{self.index_name}': {stats['total_vector_count']}")
            
            if verbose:
                print("📁 Full index stats:", stats)
            return stats

            # Note: Pinecone doesn't provide direct vector listing
            # So this only prints metadatacounts.
            #return stats
        
        except Exception as e:
            print(f"⚠️ Failed to get index stats: {e}")
            return {}

