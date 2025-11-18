import yaml
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from src.vectorstore import VectorDB
from src.utils import extract_keywords
from src.context_rules import enrich_prompt_context

class RAGChain:
    """
    Encapsulates the complete RAG logic, from retrieval to generation,
    now extended with contextual dropdown-based filtering.
    - Retrieves text, metrics, image_url, image_caption from Pinecone
    - Injects both text + image blocks into the LLM prompt
    - Uses prompts.yaml for system instructions
    """
    def __init__(self, chat_memory_history):
        """
        Initializes the RAG chain with all necessary components.
        
        Args:
            chat_memory_history: A LangChain chat message history object.
        """
        # 1. Load configuration
        with open("config/config.yaml", 'r') as f:
            config = yaml.safe_load(f)

        ## Load prompts config
        with open("config/prompts.yaml", 'r') as f:
            self.prompts_config = yaml.safe_load(f)

        # 2. Initialize components
        self.vector_db = VectorDB()
        #self.retriever = self.vector_db.as_retriever()
        self.chat_memory = chat_memory_history
        
        # LLM setup
        self.llm = ChatOpenAI(
            model=config['llm']['model_name'],
            temperature=config['llm']['temperature']
        )

        # 3. Define the prompt template
        # This prompt uses the system message from config and structures the inputs.
        self.qa_prompt = ChatPromptTemplate.from_messages([
            #("system", config['llm']['system_prompt']),
            ("system", self.prompts_config['rag_analyst_prompt']),
            ("system", self.prompts_config['copywriting_generator_prompt']),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{final_prompt}"),
        ])

        # 4. Construct the main RAG chain
        self.chain_with_history = RunnableWithMessageHistory(
            RunnablePassthrough.assign(
                #context=lambda x: self.retriever.invoke(x["final_prompt"])
                context=self.contextualized_question | self.retriever
            ) | self.qa_prompts | self.llm | StrOutputParser(),
            lambda session_id: self.chat_memory,
            input_messages_key = "final_prompt",
            history_messages_key = "chat_history",
        )


    # --------------------------------
    # Contextual Augmentation Section
    # --------------------------------
    @property
    def contextualized_question(self):
        """
        Reformulate follow-up questions to standalone questions using chat history.
        """
        system_prompt = (
            "Given a chat history and the latest user question, "
            "reformulate into a standalone question without answering. "
            "Return as-is if standalone."
        )

        contextualize_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])

        contextualize_llm = ChatOpenAI(temperature=0)
        return contextualize_prompt | contextualize_llm | StrOutputParser()

    # ------------------------------------------
    # Build Prompt with Retrieval + Filters
    # ------------------------------------------
    def build_contextual_prompt(self, user_input: str, filters: dict):
        """
        Builds a rich RAG prompt that includes:
        - Dropdown context (industry, objective, target_market)
        - High-performing retrieved ads (CTR > 0.5)
        - Visual metadata (if any)
        """

        # [1] Retrieved relevants ads from vectorstore
        results = self.vector_db.query(
            query_text=user_input,
            top_k=5,
            filters=filters,
            min_score = 0.5
        )
        """
            filters={
                "industry": filters.get("industry"),
                "objective": filters.get("campaign_objective"),
                "ctr": {"$gte": 0.5}
            },
            top_k = 5
        )
        """
        # [2] Format the retrieved context
        retrieved_context = "\n\n".join([
            f"- {r['metadata'].get('industry','')} | CTR: {r['metadata'].get('ctr','?')} | "
            f"Conv: {r['metadata'].get('conversion_rate','?')} | Text: {r['metadata'].get('text','')}"
            #f"Conv: {r['metadata'].get('industry')} | CTR: {r['metadata'].get('ctr')} | Text: {r['metadata'].get('text', '')}"
            f"{' | Image: ' + r['metadata'].get('image_url','') if r['metadata'].get('image_url') else ''}"
            #for r in results.get("matches", [])
            for r in results
        ]) if results else "No relevant dataset-backed ads found."

# [][][][][] HIGHLIGHT
        # [3] Assemble the full final prompt (for injection into LLM)
        final_prompt = f"""
User Input:
{user_input}

System Filters:
- Industry: {filters.get('industry')}
- Objective: {filters.get('campaign_objective')}
- Target Market: {filters.get('target_market')}
- Performance Threshold: CTR > 0.5
- Ads Language: {filters.get('ads_language')}
- Psychographics: {filters.get('psychographics', 'N/A')}

Relevant High-Performing Ads (from INVOKE Dataset):
{retrieved_context}

Instruction:
Generate ad ideas, headlines, copywriting, CTAs, and visual recommendations inspired by these examples.
Only use information from the dataset. If no relevant information exists, say "I don't know".
Maintain tone suitable for {filters.get('industry')} industry and optimize for {filters.get('campaign_objective')}.
"""
        return final_prompt

    # ================================
    # Inject multimodal images into LLM messages
    # ================================
    def _build_image_blocks(self, retrieved_results):
        """
        Build OpenAI multimodal image blocks:
        Each image is passed as:
        {
            "type": "image_url",
            "image_url": {
                "url": "<IMAGE_URL>"
            },
            "caption": "<CAPTION_TEXT>"
        }
        """
        image_blocks = []

        for r in retrieved_results:
            url = r["metadata"].get("image_url")
            if not url:
                continue
            
            image_blocks.append({
                "type": "image_url",
                "image_url": {"url": url}
            })

        return image_blocks
    
    # --------------------------------
    # Main Entry Point
    # --------------------------------

    def run(self, user_input: str, filters: dict = None):
        """
        Invokes the RAG chain with the user's input and manages history, and dropdown filters.
        """

        if filters is None:
            filters = {}

        # Build full contextual prompt
        final_prompt, retrieved_results = self.build_contextual_prompt(user_input, filters)

        # Step 0: Validate dropdown context (detect mismatch)
        #clarification = validate_dropdown(user_input, filters)
        #if clarification:
        #    return clarification
        
        # Step 1: Construct contextual prompt
        #contextual_prompt = self.build_contextual_prompt(user_input, filters)

        image_blocks = self._build_image_blocks(retrieved_results)

        # Step 3 -- Build multimodal message for LLM 
        multimodal_input = {
            "final_prompt": {
                "type": "input_text",
                "text": final_prompt
            },
            "images": image_blocks,
        }

        # Step 4- Run full RAG Chain, Invoke with memory
        return self.chain_with_history.invoke(
            #{"final_prompt": contextual_prompt},
            #config={"configurable": {"session_id": "default_session"}}
            #{"input": user_input, "final_prompt": final_prompt},
            multimodal_input,
            config={"configurable": {"session_id": "default_session"}}
        )
    
    # ------------------
    # Helper
    # ------------------
    @property
    def retriever(self):
        return self.vector_db.as_retriever(search_kwargs={"k":5})
    
    @staticmethod
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
