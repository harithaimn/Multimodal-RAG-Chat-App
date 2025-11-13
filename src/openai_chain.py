import yaml
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from src.vectorstore import VectorDB
from src.utils import extract_keywords
from src.context_rules import validate_dropdown


class RAGChain:
    """
    Encapsulates the complete RAG logic, from retrieval to generation,
    now extended with contextual dropdown-based filtering.
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
            prompts_config = yaml.safe_load(f)

        # 2. Initialize components
        self.vector_db = VectorDB()
        self.retriever = self.vector_db.as_retriever()
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
            ("system", prompts_config['rag_analyst_prompt']),
            ("system", prompts_config['copywriting_generator_prompt']),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{final_prompt}"),
        ])

        # 4. Construct the main RAG chain
        self.chain_with_history = RunnableWithMessageHistory(
            RunnablePassthrough.assign(
                context=lambda x: self.retriever.invoke(x["final_prompt"])
                #context=self.contextualized_question | self.retriever
            ) | self.qa_prompts | self.llm | StrOutputParser(),
            lambda session_id: self.chat_memory,
            input_messages_key = "final_prompt",
            history_messages_key = "chat_history",
        )


    # --------------------------------
    # Contextual Augmentation Section
    # --------------------------------

    def build_contextual_prompt(self, user_input: str, filters: dict):
        """
        Builds a rich RAG prompt that includes:
        - Dropdown context (industry, objective, target_market)
        - High-performing retrieved ads (CTR > 0.5)
        """

        # [1] Retrieved relevants ads from vectorstore
        results = self.vector_db.query(
            query=user_input,
            filters={
                "industry": filters.get("industry"),
                "objective": filters.get("campaign_objective"),
                "ctr": {"$gte": 0.5}
            },
            top_k = 5
        )

        # [2] Format the retrieved context
        retrieved_context = "\n\n".join([
            f"- {r['metadata'].get('industry')} | CTR: {r['metadata'].get('ctr')} | Text: {r['metadata'].get('text', '')}"
            for r in results.get("matches", [])
        ]) if results else "No similar ads found."

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

Relevant High-Performing Ads (from INVOKE Dataset):
{retrieved_context}

Instruction:
Generate ad ideas, headlines, copywriting, CTAs and all others inspired by these examples.
Maintain the tone suitable for {filters.get('industry')} industry and optimize for {filters.get('campaign_objective')}.
        """

        return final_prompt

    # ---------------------------------
    # Base Methods
    # ---------------------------------

    @staticmethod
    def format_docs(docs):
        """Helper function to format retrieved documents into a single string."""
        return "\n\n".join(doc.page_content for doc in docs)

    @property
    def contextualized_question(self):
        """
        Creates a sub-chain to rephrase a follow-up question into a standalone question
        using the conversation history.
        """
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ])
        contextualize_q_llm = ChatOpenAI(temperature=0)
        return contextualize_q_prompt | contextualize_q_llm | StrOutputParser()

    # --------------------------------
    # Main Entry Point
    # --------------------------------

    def run(self, user_input: str, filters: dict = None):
        """
        Invokes the RAG chain with the user's input and manages history, and dropdown filters.
        """

        if filters is None:
            filters = {}

        # Step 0: Validate dropdown context (detect mismatch)
        clarification = validate_dropdown(user_input, filters)
        if clarification:
            return clarification
        
        # Step 1: Construct contextual prompt
        contextual_prompt = self.build_contextual_prompt(user_input, filters)

        # Step 2L Run full RAG Chain
        return self.chain_with_history.invoke(
            {"final_prompt": contextual_prompt},
            config={"configurable": {"session_id": "default_session"}}
        )