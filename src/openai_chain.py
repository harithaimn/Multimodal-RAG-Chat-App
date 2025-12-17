# src/openai_chain.py

"""
Multimodal RAG pipeline (POC-ready)

Responsibilities:
- Retrieve relevant ads from Pinecone
- Build dataset-grounded prompt
- Inject image URLs as multimodal context
- Call OpenAI chat model for final generation

Design:
- No LangChain memory abstractions
- No inferred attributes
- Dataset-first grounding only
"""

import yaml
from typing import Dict, List

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from src.vectorstore import VectorDB


# -------------------------------------------------
# Load configuration
# -------------------------------------------------

with open("config/config.yaml", "r") as f:
    CONFIG = yaml.safe_load(f)

with open("config/prompts.yaml", "r") as f:
    PROMPTS = yaml.safe_load(f)

LLM_MODEL = CONFIG["llm"]["model_name"]
LLM_TEMP = CONFIG["llm"]["temperature"]
EMBEDDING_MODEL = CONFIG["embedding_model"]["model_name"]


# -------------------------------------------------
# RAG Chain
# -------------------------------------------------

class RAGChain:
    def __init__(self, chat_history: List[Dict] | None = None):
        """
        chat_history: Streamlit session chat history (list of {role, content})
        """
        self.vector_db = VectorDB(embedding_model=EMBEDDING_MODEL)
        self.chat_history = chat_history or []

        self.llm = ChatOpenAI(
            model=LLM_MODEL,
            temperature=LLM_TEMP,
        )

    # -------------------------------------------------
    # Retrieval
    # -------------------------------------------------

    def _retrieve(self, user_input: str, filters: Dict, k: int = 5):
        retriever = self.vector_db.as_retriever()
        return retriever(user_input, filters=filters, k=k)

    # -------------------------------------------------
    # Prompt construction
    # -------------------------------------------------

    def _build_retrieved_block(self, retrieved: List[Dict]) -> str:
        if not retrieved:
            return "No relevant examples found in the dataset."

        lines = []
        for i, r in enumerate(retrieved, start=1):
            md = r.get("metadata", {})
            lines.append(
                f"Reference #{i}: "
                f"Campaign='{md.get('campaign_name','')}', "
                f"CTR={md.get('ctr','?')}, "
                f"Caption='{md.get('caption','')}'"
            )
        return "\n".join(lines)

    def _build_prompt(self, user_input: str, filters: Dict, retrieved: List[Dict]) -> str:
        retrieved_block = self._build_retrieved_block(retrieved)

        return f"""
User Question:
{user_input}

Applied Filters:
{filters}

Retrieved Dataset Examples:
{retrieved_block}

Task:
Using ONLY the retrieved examples above:
- Generate insights
- Suggest ad copy ideas
- Recommend visuals

Rules:
- Do NOT hallucinate facts
- Do NOT invent metrics or audiences
- If information is missing, say so explicitly
"""

    # -------------------------------------------------
    # Multimodal image blocks
    # -------------------------------------------------

    # def _build_image_blocks(self, retrieved: List[Dict]) -> List[Dict]:
    #     blocks = []
    #     BLOCKED_DOMAINS = ("fbcdn.net", "facebook.com")

    #     for r in retrieved:
    #         img_url = r.get("image_url")
    #         if not img_url:
    #             continue

    #         if any(d in img_url for d in BLOCKED_DOMAINS):
    #             continue  # skip unsafe image URLs

    #         blocks.append({
    #             "type": "image_url",
    #             "image_url": {"url": img_url}
    #         })

    #     return blocks
    def _build_image_blocks(self, retrieved: List[Dict]) -> List[Dict]:
        # Multimodal disabled — image URLs from Meta/IG are not fetchable by OpenAI
        return []

    # -------------------------------------------------
    # Main execution
    # -------------------------------------------------

    def run(self, user_input: str, filters: Dict | None = None):
        """
        Returns:
            answer_text (str)
            retrieved_results (list)
        """
        filters = filters or {}

        # 1. Retrieve
        retrieved = self._retrieve(user_input, filters)

        # 2. Build prompt
        final_prompt = self._build_prompt(user_input, filters, retrieved)

        # 3. Build multimodal human message
        image_blocks = self._build_image_blocks(retrieved)

        human_content = [
            {"type": "text", "text": final_prompt},
            *image_blocks
        ]

        messages = [
            SystemMessage(content=PROMPTS.get("system_guardrails", "")),
            #HumanMessage(content=human_content)
            HumanMessage(content=final_prompt)
        ]

        # 4. Invoke LLM
        response = self.llm.invoke(messages)

        # 5. Extract text safely
        answer_text = getattr(response, "content", str(response))

        return answer_text, retrieved
