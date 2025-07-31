# rag_core.py
import json
import logging
import time
import re
from typing import Dict, List, Tuple, Any, Optional
from difflib import SequenceMatcher

# Import components
import config
from llm_interface import LLMInterface
from vector_store import VectorStore
from evaluation import evaluate_response, assess_answer_confidence, record_feedback # Assuming this has the correct signature
from utils import format_reasoning_trace, generate_hop_graph
from graph_db import Neo4jGraphDB # Import the graph DB class


class CNTRagSystem:
    """
    Core RAG system orchestrating multi-hop reasoning for CNT data.
    """
    def __init__(self,
                 llm_interface: LLMInterface,
                 vector_store: VectorStore,
                 graph_db: Neo4jGraphDB, # Add graph_db to init
                 logger: logging.Logger,
                 feedback_db_path: str,
                 feedback_history: List[Dict[str, Any]], # Pass loaded history
                 user_type: str = "advanced", # Added user_type, default to novice
                 graph_dir: str = config.DEFAULT_GRAPH_DIR,
                 # Context management params from config
                 max_context_tokens: int = config.MAX_CONTEXT_TOKENS,
                 char_to_token_ratio: float = config.CHAR_TO_TOKEN_RATIO,
                 similarity_threshold: float = config.SIMILARITY_THRESHOLD
                 ):
        """
        Initialize the CNTRagSystem.

        Args:
            llm_interface: Instance of LLMInterface.
            vector_store: Instance of a VectorStore implementation.
            logger: Logger instance.
            feedback_db_path: Path to save feedback data.
            feedback_history: Pre-loaded feedback history list.
            user_type: Type of user, "novice" or "advanced".
            graph_dir: Directory to save reasoning graphs.
            max_context_tokens: Max estimated tokens for context window.
            char_to_token_ratio: Estimated chars per token.
            similarity_threshold: Threshold for context deduplication.
        """
        self.llm_interface = llm_interface
        self.vector_store = vector_store
        self.graph_db = graph_db # Store the graph_db instance
        self.logger = logger
        self.feedback_db_path = feedback_db_path
        self.feedback_history = feedback_history # Use the passed list
        self.user_type = 'advanced'
        self.graph_dir = graph_dir
        self.max_context_tokens = max_context_tokens
        self.char_to_token_ratio = char_to_token_ratio
        self.similarity_threshold = similarity_threshold

        if not self.vector_store.is_ready():
            self.logger.critical("Vector store provided is not ready (loaded or built). RAG system may fail.")

    # --- Query Processing Steps ---
    def _transform_query(self, query: str) -> str:
        """Basic query transformation (lowercase)."""
        self.logger.debug(f"Applying query transformations to: '{query}'")
        transformed_query = query.lower()
        if transformed_query != query:
            self.logger.info(f"Query transformed: '{query}' -> '{transformed_query}'")
        return transformed_query
    
    def generate_contextual_query(self, chat_history: List[Dict[str, Any]], new_question: str) -> str:
        """
        Generates a standalone query from the chat history and the new question.
        """
        self.logger.info("Generating standalone query from chat history...")
        
        # If history is empty or this is the first real question, no need to rephrase
        if len(chat_history) <= 1:
            self.logger.info("No significant history found. Using new question as is.")
            return new_question

        # Format the chat history for the prompt
        formatted_history = ""
        for message in chat_history[:-1]: # Exclude the latest question
            role = "User" if message["role"] == "user" else "Assistant"
            formatted_history += f"{role}: {message['content']}\n"
        
        prompt = f"""Given the following chat history and a new question, rephrase the new question to be a standalone question that can be understood without the preceding context.
        If the new question is already a complete, standalone question, you can return it as is.
        Do not answer the question, just reformulate it.

        **Chat History:**
        {formatted_history}

        **New Question:**
        "{new_question}"

        **Standalone Question:**
        """
        
        rephrased_query = self.llm_interface.generate_response(prompt)

        if rephrased_query.startswith("LLM_ERROR"):
            self.logger.warning(f"Failed to rephrase query due to LLM error: {rephrased_query}. Using original question.")
            return new_question
        
        # Simple cleanup of the LLM's output
        rephrased_query = rephrased_query.strip().strip('"')

        self.logger.info(f"Original question: '{new_question}' -> Standalone question: '{rephrased_query}'")
        return rephrased_query

    def _expand_query(self, query: str, num_expansions: int = 2) -> List[str]:
        """Generate query variations using the LLM based on user type."""
        if num_expansions <= 0:
            return [query]

        self.logger.info(f"Expanding query '{query[:100]}...' for '{self.user_type}' user into {num_expansions} variations.")
        
        common_instructions = f"""
        Original Question:
        "{query}"

        Instructions for alternative queries:
        1.  Understand the core scientific intent of the original question.
        2.  Create {num_expansions} distinct alternative queries.
        3.  Use relevant scientific terms (e.g., 'CVD synthesis', 'catalyst', 'chirality', 'Raman spectroscopy', 'SWCNT', 'MWCNT', 'functionalization', specific chemical formulas if relevant).
        4.  Focus on experimental parameters, material properties, characterization techniques, or applications mentioned in CNT research.
        5.  Return ONLY the alternative queries, one per line. Do not include numbering, explanations, or the original query.
        """

        if self.user_type == "novice":
            expansion_prompt = f"""You are a language expert helping a beginner understand Carbon Nanotubes (CNTs).
            Generate {num_expansions} alternative search queries for the following original question.
            The goal is to find information that can be explained simply. The queries should target fundamental concepts and clear examples.
            Focus on CNT-related factors such as diameter, chirality, basic electronic properties, common functionalization techniques, and general synthesis conditions.
            Use clear phrasing for retrieval of easy-to-understand documents.
            {common_instructions}
            Alternative Queries (for a novice):"""
        elif self.user_type == "advanced":
            expansion_prompt = f"""You are an expert search query generator assisting an advanced researcher in Carbon Nanotubes (CNTs).
            Generate {num_expansions} alternative search queries for the following original question.
            The queries should emphasize critical experimental conditions (exact temperature, catalyst details, thickness), detailed growth trends, stability, structural quality, and nuances of CNT synthesis and properties.
            Use precise, technical phrasing to retrieve in-depth scientific documents.
            {common_instructions}
            Alternative Queries (for an advanced researcher):"""
        else: # Default (should not happen due to __init__ check)
            self.logger.warning("User type not properly set for query expansion, using default.")
            expansion_prompt = f"""You are an expert search query generator specializing in scientific literature about Carbon Nanotubes (CNTs).
            Generate {num_expansions} alternative search queries for the following original question.
            The goal is to capture the same information need using different phrasing, keywords, and potential concepts found in CNT research papers.
            {common_instructions}
            Alternative Queries:"""

        response_text = self.llm_interface.generate_response(expansion_prompt)

        if response_text.startswith("LLM_ERROR"):
            self.logger.warning(f"Query expansion failed: {response_text}")
            return [query]

        expanded_queries = [q.strip() for q in response_text.split('\n') if q.strip()]
        expanded_queries = expanded_queries[:num_expansions] # Limit to requested number

        all_queries = [query] + expanded_queries
        unique_queries = list(dict.fromkeys(all_queries))

        self.logger.info(f"Generated {len(unique_queries)} unique queries: {unique_queries}")
        return unique_queries

    def _reason_and_refine_query(self, original_query: str, accumulated_context: str, history: List[str], current_hop: int, retrieval_failed_last: bool = False) -> Tuple[str, str]:
        """Uses the LLM to decide the next step based on user type."""
        self.logger.info(f"--- Hop {current_hop} Reasoning for '{self.user_type}' user ---")
        context_snippet = accumulated_context.strip()[:500]
        self.logger.debug(f"Reasoning based on History: {' -> '.join(history)}")
        self.logger.debug(f"Context Snippet Provided:\n{context_snippet if context_snippet else 'None'}...\n")
        if retrieval_failed_last:
             self.logger.warning(f"Reasoning invoked after retrieval failure in hop {current_hop}.")
        common_reasoning_intro = f"""
        Your SOLE task is to determine the *next action* based on the Original Question and the Accumulated Context retrieved so far.

        Original Question:
        "{original_query}"

        Accumulated Context (from previous search steps):
        --- START CONTEXT ---
        {accumulated_context if accumulated_context.strip() else "No context retrieved yet."}
        --- END CONTEXT ---

        Reasoning History (Previous search queries used):
        --- START HISTORY ---
        {' -> '.join(history) if history else "This is the first reasoning step."}
        --- END HISTORY ---
        """

        common_reasoning_critical = """
        CRITICAL:
        - Perform the reasoning steps above internally.
        - Your final output MUST still be only ONE line: EITHER "ANSWER_COMPLETE" OR "NEXT_QUERY: [your query]".
        - NO explanations in your output.
        """
        
        prompt = f"""You are an expert reasoning engine for a RAG system assisting an *advanced researcher* with questions about Carbon Nanotube (CNT) research.
        The user is looking for detailed technical insights, numerical trends, and potential optimizations.
        {common_reasoning_intro}
        Instructions:
        1. Analyze if the Accumulated Context *comprehensively and technically* answers the Original Question for an advanced researcher.
            Focus on key findings related to CNT synthesis, including the influence of factors like temperature (e.g., 604°C), catalyst type (e.g., Fe or Aluminum Oxide), and thickness (e.g., 0.61 µm) on growth rates, stability, and maximum height.
            Look for advanced observations such as growth stabilization times (e.g., 1118 seconds), numerical trends, and any anomalies in the data.
        2. Consider the search queries used so far.
        3. Decide the single next action:
            a. If context fully and technically answers: Output *exactly*: ANSWER_COMPLETE
            b. If context is insufficient or more depth is needed for an advanced user, identify the *single most critical missing piece of technical information* or *next logical sub-question for deeper analysis*.
                Formulate a *concise, specific scientific search query* for this, potentially focusing on optimizations or experimental modifications to refine CNT synthesis.
                Output *exactly* in this format: NEXT_QUERY: [Your concise technical query here]
            c. If retrieval errors occurred or context is irrelevant, output: ANSWER_COMPLETE

        Reasoning Steps (Think Internally before deciding):
        1. What core technical information, numerical data, or optimization insights are needed for an advanced user regarding the Original Question?
        2. Does the Accumulated Context contain this information with sufficient depth and detail? List supporting evidence/snippets.
        3. Are there specific, critical technical gaps or areas for further detailed exploration? If so, what are they?
        4. Based on these gaps, what is the most effective *next* technical query to deepen the understanding or explore optimizations, or is the answer complete?
        {common_reasoning_critical}
        Decision:"""
        
        response_text = self.llm_interface.generate_response(prompt)

        action = "ERROR"; value = f"LLM response parsing failed. Raw: '{response_text}'"
        if response_text.startswith("LLM_ERROR"):
             self.logger.error(f"Reasoning failed due to LLM error: {response_text}")
             value = response_text; action = "ANSWER_COMPLETE" 
             self.logger.warning("LLM error during reasoning. Assuming ANSWER_COMPLETE.")
        else:
            response_clean = response_text.strip()
            if re.fullmatch(r"ANSWER_COMPLETE", response_clean, re.IGNORECASE):
                action = "ANSWER_COMPLETE"; value = ""
                self.logger.info(">>> Reasoning Decision: ANSWER_COMPLETE")
            else:
                match_next = re.fullmatch(r"NEXT_QUERY:\s*(.+)", response_clean, re.IGNORECASE | re.DOTALL)
                if match_next:
                    new_query = match_next.group(1).strip()
                    if new_query:
                        action = "NEXT_QUERY"; value = new_query
                        self.logger.info(f">>> Reasoning Decision: NEXT_QUERY -> '{new_query}'")
                    else:
                        action = "ANSWER_COMPLETE"; value = ""
                        self.logger.warning("LLM generated NEXT_QUERY but query was empty. Treating as ANSWER_COMPLETE.")
                else:
                    action = "ANSWER_COMPLETE"; value = "" 
                    self.logger.warning(f"Unexpected reasoning response format: '{response_clean}'. Defaulting to ANSWER_COMPLETE.")
        return action, value

    def _format_context_from_chunks(self, chunks: List[Dict[str, Any]]) -> str:
        """Formats the retrieved chunks into a string for the LLM prompt, including detailed source."""
        if not chunks:
             return "No relevant context found."

        context_parts = []
        for chunk_dict in chunks:
            metadata = chunk_dict.get('metadata', {})
            score = chunk_dict.get('score', 0.0)
            
            # Construct a detailed, parsable source string for the LLM
            source_info_str = (
                f"[Source: Doc: '{metadata.get('document_name', 'N/A')}', "
                f"Page: {metadata.get('page_number', 'N/A')}, "
                f"ChunkID: '{metadata.get('chunk_id', 'N/A')}', "
                f"Score: {score:.4f}]"
            )
            
            context_parts.append(f"{source_info_str}\n{chunk_dict.get('text', '')}")

        return "\n\n---\n\n".join(context_parts)
    
    def _summarize_text_block(self, text_to_summarize: str, original_query_for_context: Optional[str] = None) -> str:
        if not text_to_summarize.strip():
            return ""
        self.logger.info(f"Summarizing text block (length: {len(text_to_summarize)})...")
        query_guidance = f"Focus on details relevant to the original question: '{original_query_for_context}'" if original_query_for_context else ""
        summary_prompt = f"""Summarize the following context section from scientific literature or experimental data regarding Carbon Nanotubes (CNTs).
        The user asking the question is considered '{self.user_type}'.
        Preserve all key scientific findings, experimental parameters (temperature, pressure, catalyst, etc.), numerical results, material properties, characterization data (e.g., Raman shifts, diameters), and specific CNT types (SWCNT, MWCNT, chirality) mentioned.
        {query_guidance}
        Be concise but retain the essential scientific information needed to answer questions based on this section.

        Context Section to Summarize:
        ---
        {text_to_summarize[:config.MAX_CONTEXT_TOKENS * 2]}
        ---

        Concise Scientific Summary:"""
        summarized = self.llm_interface.generate_response(summary_prompt)
        if summarized.startswith("LLM_ERROR"):
            self.logger.error(f"Failed to summarize text block: {summarized}. Returning original (potentially truncated).")
            return text_to_summarize[:self.max_context_tokens // 2]
        else:
            self.logger.info("Summarization of text block complete.")
            return summarized

    def _manage_context(self,
                        accumulated_context_list: List[str],
                        new_chunk_list: List[Dict[str, Any]],
                        original_query: str
                       ) -> Tuple[List[str], List[Dict[str, Any]]]:
        if not new_chunk_list:
            self.logger.debug("No new chunks to manage.")
            return accumulated_context_list, []

        unique_new_chunks_by_text: Dict[str, Dict[str, Any]] = {}
        for chunk in new_chunk_list:
            text = chunk.get('text', '')
            if not text: continue
            if text not in unique_new_chunks_by_text or chunk.get('score', 0) > unique_new_chunks_by_text[text].get('score', 0):
                 unique_new_chunks_by_text[text] = chunk
        deduped_new_chunks = list(unique_new_chunks_by_text.values())

        existing_texts = set()
        for ctx_str in accumulated_context_list:
             text_content = ctx_str
             match = re.search(r'\]\s*(.*)', ctx_str, re.DOTALL | re.IGNORECASE) or re.search(r'\]:\s*\n?(.*)', ctx_str, re.DOTALL | re.IGNORECASE)
             if match: text_content = match.group(1).strip()
             else: text_content = ctx_str.strip()
             if text_content.startswith(("[SUMMARIZED CONTEXT]:", "[SUMMARIZED HOP CONTEXT]:")):
                 text_content = re.sub(r"^\[SUMMARIZED (?:HOP )?CONTEXT\]:\s*", "", text_content, flags=re.IGNORECASE).strip()
             existing_texts.add(text_content)

        truly_new_chunks_list_of_dicts = []
        for chunk_dict in deduped_new_chunks:
            text_to_check = chunk_dict.get('text', '')
            is_similar = False
            for existing_text in existing_texts:
                 if text_to_check == existing_text or SequenceMatcher(None, text_to_check, existing_text).ratio() > self.similarity_threshold:
                     is_similar = True
                     self.logger.debug("Found new chunk identical or highly similar to existing context. Skipping.")
                     break
            if not is_similar:
                 truly_new_chunks_list_of_dicts.append(chunk_dict)
                 existing_texts.add(text_to_check)

        if not truly_new_chunks_list_of_dicts:
            self.logger.info("No truly new, unique context chunks found in this hop after deduplication.")
            return accumulated_context_list, []

        current_context_list = list(accumulated_context_list)
        new_context_string_for_this_hop = self._format_context_from_chunks(truly_new_chunks_list_of_dicts)

        if new_context_string_for_this_hop.strip():
            if self.llm_interface.llm_provider != "google":
                self.logger.info(f"LLM Provider is '{self.llm_interface.llm_provider}'. Summarizing new context from this hop...")
                summarized_new_context = self._summarize_text_block(new_context_string_for_this_hop, original_query)
                if summarized_new_context.strip():
                    current_context_list.append(f"[SUMMARIZED HOP CONTEXT]:\n{summarized_new_context}")
                    self.logger.info(f"Added summarized hop context. Current list length: {len(current_context_list)}")
                else:
                    self.logger.warning("Summarization of new hop context resulted in empty string. Adding original formatted chunks for this hop instead.")
                    current_context_list.append(new_context_string_for_this_hop)
                    self.logger.info(f"Added original (non-summarized) hop context. Current list length: {len(current_context_list)}")
            else:
                self.logger.info(f"LLM Provider is 'google'. Adding raw formatted chunks from this hop.")
                current_context_list.append(new_context_string_for_this_hop)
                self.logger.info(f"Added raw hop context for Google. Current list length: {len(current_context_list)}")
        else:
            self.logger.debug("Formatted new context string for this hop is empty. Not adding to context list.")

        estimated_tokens = sum(len(ctx) for ctx in current_context_list) / self.char_to_token_ratio
        self.logger.debug(f"Context size before *overall* summarization: ~{estimated_tokens:.0f} tokens. List length: {len(current_context_list)}")

        while estimated_tokens > self.max_context_tokens and len(current_context_list) > 1:
             self.logger.info(f"Overall context too large ({estimated_tokens:.0f} > {self.max_context_tokens} tokens), summarizing oldest overall context block...")
             oldest_context_block = current_context_list.pop(0)
             if oldest_context_block.strip().startswith("[SUMMARIZED CONTEXT]:"):
                 self.logger.warning("Oldest context block is already an *overall* summary. Keeping it.")
                 current_context_list.insert(0, oldest_context_block)
                 break
             summarized_oldest_block = self._summarize_text_block(oldest_context_block, original_query)
             if not summarized_oldest_block.strip():
                  self.logger.error("Summarization of oldest block yielded empty string. Discarding block.")
             else:
                  current_context_list.insert(0, f"[SUMMARIZED CONTEXT]:\n{summarized_oldest_block}")
                  self.logger.info("Overall summarization of oldest block complete.")
             estimated_tokens = sum(len(ctx) for ctx in current_context_list) / self.char_to_token_ratio
             self.logger.info(f"Context size after *overall* summarization of oldest: ~{estimated_tokens:.0f} tokens")

        if estimated_tokens > self.max_context_tokens:
            self.logger.warning(f"Final context size (~{estimated_tokens:.0f}) still exceeds max_tokens ({self.max_context_tokens}) after all summarization attempts.")

        return current_context_list, truly_new_chunks_list_of_dicts

    def _generate_final_answer(self, original_query: str, accumulated_context_list: List[str]) -> str:
        """Generates the final answer, prompting the LLM to cite its sources."""
        self.logger.info(f"Generating final answer for '{self.user_type}' user from accumulated context...")
        final_context_str = "\n\n==== CONTEXT FROM HOP/SUMMARY SEPARATOR ====\n\n".join(accumulated_context_list)

        if not final_context_str.strip() or final_context_str.strip() == "No relevant context found.":
            self.logger.warning("No context available for final answer generation.")
            return "Based on the retrieved information, a detailed analysis could not be performed due to insufficient context."

        # This prompt is the same as before and works perfectly for this goal
        prompt = f"""You are an expert researcher synthesizing a detailed technical analysis for an advanced colleague on Carbon Nanotubes (CNTs).

        **Original Question:**
        "{original_query}"

        **Accumulated Context:**
        Use ONLY the following context to answer the question. Each piece of context is preceded by its source information in the format [Source: Doc: '...', Page: ..., ChunkID: '...'].
        --- START CONTEXT ---
        {final_context_str}
        --- END CONTEXT ---

        **Your Task:**
        1.  **Synthesize and Structure:** Write a comprehensive, well-structured answer to the "Original Question". Organize your answer into logical sections with clear, bolded headings (e.g., using Markdown like **Heading Title**). Explain concepts and their relationships.
        2.  **Cite Everything:** You MUST cite every piece of information you use from the context. Place the citation at the end of the sentence or clause that uses the information.
        3.  **Citation Format:** The citation format MUST BE: `(Source: Doc: '[Doc]', Page: [Page], ChunkID: '[ChunkID]')`.
        4.  **Strictly Adhere to Context:** Base your entire answer ONLY on the "Accumulated Context".
        5.  **Identify Gaps:** After providing the main answer, add a final section titled "**Missing Information**". In this section, describe what specific technical details are needed but are not present in the provided context.

        ---
        **Final Synthesized Answer (Detailed, Structured, and with In-Text Citations):**
        """
        
        final_response = self.llm_interface.generate_response(prompt)

        if final_response.startswith("LLM_ERROR"):
             self.logger.error(f"Final answer generation failed: {final_response}")
             return f"I encountered an error while trying to generate the final answer ({final_response})."
        else:
             self.logger.info(f"Final answer generated (length: {len(final_response)}).")
             final_response = re.sub(r"^\s*Final Synthesized Answer:?.*?\s*", "", final_response, flags=re.IGNORECASE | re.DOTALL).strip()
             return final_response

    # --- NEW: Query Planner Function ---
    def _plan_query(self, query: str) -> Dict[str, Any]:
        """
        Uses an LLM to deconstruct a query into a semantic query and metadata filters.
        """
        prompt = f"""
        You are an expert in Carbon Nanotube research. Analyze the user's query and break it down into a structured search plan.
        The available entity types for filtering are: "Method", "Catalyst", "Substrate", "CNT_Type", "Carbon_Source".

        User Query: "{query}"

        Based on the query, generate a JSON object with two keys:
        1. "semantic_query": A rephrased query for vector search, focusing on the core concepts.
        2. "metadata_filters": A dictionary where keys are entity types and values are entity names. Only include filters explicitly mentioned.

        Example 1:
        User Query: "How does fixed catalyst iron CVD impact MWCNT growth on silicon?"
        Your JSON output:
        {{
            "semantic_query": "growth dynamics and mechanisms of multi-walled carbon nanotubes using iron fixed catalyst CVD on silicon wafers",
            "metadata_filters": {{
                "Method": "CVD",
                "Catalyst": "Iron",
                "CNT_Type": "MWCNT",
                "Substrate": "Silicon"
            }}
        }}

        Example 2:
        User Query: "Tell me about growing carbon nanotubes."
        Your JSON output:
        {{
            "semantic_query": "general principles and techniques for growing carbon nanotubes",
            "metadata_filters": {{}}
        }}

        Now, analyze the following query and produce only the JSON output.

        User Query: "{query}"
        JSON Output:
        """
        response_str = self.llm_interface.generate_response(prompt)
        try:
            response_str = response_str.strip().replace("```json", "").replace("```", "").strip()
            plan = json.loads(response_str)
            if isinstance(plan, dict) and "semantic_query" in plan and "metadata_filters" in plan:
                return plan
        except (json.JSONDecodeError, TypeError):
            self.logger.warning(f"Failed to decode query plan. Using query as is. Raw: {response_str[:100]}...")
        # Fallback plan
        return {"semantic_query": query, "metadata_filters": {}}
    
    def process_query(self, question: str, top_k: int = config.DEFAULT_TOP_K, max_hops: int = config.DEFAULT_MAX_HOPS,
                        use_query_expansion: bool = False,
                        request_evaluation: bool = True,
                        generate_graph: bool = True,
                        record_user_feedback: Optional[Dict[str, Any]] = None
                        ) -> Dict[str, Any]:
        process_start_time = time.time()
        self.logger.info(f"\n--- Starting CNT Query Process for '{self.user_type}' user ---")
        self.logger.info(f"Original Question: '{question}'")
        self.logger.info(f"Config: top_k={top_k}, max_hops={max_hops}, expansion={use_query_expansion}, eval={request_evaluation}, graph={generate_graph}")

        if not self.vector_store.is_ready():
            self.logger.critical("Vector database is not ready. Cannot process query.")
            return {"final_answer": "Error: Knowledge base not available.", "source_info": "Initialization Error", "reasoning_trace": [], "formatted_reasoning": "Vector DB not ready", "confidence_score": None, "evaluation_metrics": None, "debug_info": {}}

        original_query = question
        current_query = self._transform_query(original_query)

        accumulated_context_list: List[str] = []
        reasoning_trace: List[str] = ["START"]
        search_query_history: List[str] = [current_query]
        hops_taken: int = 0
        graph_query_history: List[str] = [current_query]
        action: Optional[str] = None
        final_context_sources: List[Dict[str, Any]] = []
        unique_chunk_ids_in_context = set()

        if use_query_expansion:
            self.logger.info("Applying initial query expansion...")
            expanded_queries = self._expand_query(current_query, num_expansions=2)
            if expanded_queries:
                current_query = expanded_queries[0]
                search_query_history = list(dict.fromkeys(expanded_queries))
                graph_query_history = list(dict.fromkeys(expanded_queries))
                reasoning_trace.append(f"Initial Expanded Queries: {expanded_queries}")

        for hop in range(max_hops):
            hops_taken = hop + 1
            self.logger.info(f"\n--- Hop {hops_taken}/{max_hops} ---")
            reasoning_trace.append(f"--- Hop {hops_taken} ---")

            if not current_query:
                self.logger.warning(f"Hop {hops_taken}: Query is empty. Breaking.")
                reasoning_trace.append(f"Hop {hops_taken}: Error - Query empty.")
                break

            # --- START OF CORRECTED HYBRID SEARCH LOGIC ---
            retrieval_succeeded_this_hop = False
            retrieved_chunks: List[Dict[str, Any]] = []
            
            # 1. Plan the query to get semantic part and metadata filters
            search_plan = self._plan_query(current_query)
            semantic_query = search_plan.get("semantic_query", current_query)
            metadata_filters = search_plan.get("metadata_filters", {})
            reasoning_trace.append(f"Hop {hops_taken}: Planned Query -> Semantic='{semantic_query[:100]}...', Filters={metadata_filters}")

            # 2. Get candidate chunk IDs from the KG if filters exist
            candidate_chunk_ids = []
            if metadata_filters:
                candidate_chunk_ids = self.graph_db.get_chunk_ids_for_entities(metadata_filters)
                if not candidate_chunk_ids:
                    self.logger.warning(f"KG found no chunks matching filters {metadata_filters}. Proceeding with semantic search on all data.")
                    reasoning_trace.append(f"Hop {hops_taken}: KG found no matching chunks for filters.")
                else:
                    reasoning_trace.append(f"Hop {hops_taken}: KG pre-filtered to {len(candidate_chunk_ids)} candidate chunks.")

            # 3. Perform vector search using the SEMANTIC query
            self.logger.info(f"Retrieving context for semantic query: '{semantic_query}'")
            query_embedding = self.llm_interface.get_embedding(semantic_query, task_type="RETRIEVAL_QUERY")

            if query_embedding is None:
                self.logger.error("Query embedding failed. Cannot retrieve.")
                reasoning_trace.append(f"Hop {hops_taken}: Query embedding failed.")
            else:
                # Retrieve a larger pool of candidates from the vector store
                all_retrieved_chunks = self.vector_store.query(query_embedding, top_k=top_k * 5)

                if candidate_chunk_ids:
                    # Filter the vector results by the KG candidates
                    final_filtered_chunks = [
                        chunk for chunk in all_retrieved_chunks if chunk.get("id") in candidate_chunk_ids
                    ]
                    retrieved_chunks = final_filtered_chunks[:top_k]
                    self.logger.info(f"Post-filtered {len(all_retrieved_chunks)} vector results down to {len(retrieved_chunks)} based on KG.")
                else:
                    # No KG filters, so just use the top vector results
                    retrieved_chunks = all_retrieved_chunks[:top_k]

                if not retrieved_chunks:
                    self.logger.warning(f"Hybrid retrieval returned no relevant chunks for hop {hops_taken}.")
                    reasoning_trace.append(f"Hop {hops_taken}: Hybrid retrieval yielded no results.")
                else:
                    self.logger.info(f"Hybrid retrieval found {len(retrieved_chunks)} chunks for hop {hops_taken}.")
                    top_score_str = f"{retrieved_chunks[0]['score']:.4f}" if retrieved_chunks else "N/A"
                    reasoning_trace.append(f"Hop {hops_taken}: Retrieved {len(retrieved_chunks)} chunk(s). Top score: {top_score_str}")
                    retrieval_succeeded_this_hop = True
            # --- END OF CORRECTED HYBRID SEARCH LOGIC ---

            accumulated_context_list, added_unique_chunks_dicts = self._manage_context(
                accumulated_context_list,
                retrieved_chunks,
                original_query
            )

            if added_unique_chunks_dicts:
                reasoning_trace.append(f"Hop {hops_taken}: Processed {len(added_unique_chunks_dicts)} unique chunks for context.")
                for chunk_data in added_unique_chunks_dicts:
                    chunk_id = chunk_data.get('id')
                    if chunk_id and chunk_id not in unique_chunk_ids_in_context:
                        # NOTE: final_context_sources is the correct variable to populate now
                        final_context_sources.append({
                            "document_name": chunk_data.get("metadata", {}).get("document_name", "N/A"),
                            "page_number": chunk_data.get("metadata", {}).get("page_number", "N/A"),
                            "chunk_id": chunk_id,
                            "retrieval_score": chunk_data.get("score", 0.0),
                            "text_snippet": chunk_data.get("text", "")[:150] + "..."
                        })
                        unique_chunk_ids_in_context.add(chunk_id)
            elif retrieved_chunks:
                reasoning_trace.append(f"Hop {hops_taken}: No new unique chunks added (duplicates/similar).")

            full_context_for_reasoning = "\n\n==== CONTEXT FROM HOP/SUMMARY SEPARATOR ====\n\n".join(accumulated_context_list)
            
            action, value = self._reason_and_refine_query(
                original_query,
                full_context_for_reasoning,
                graph_query_history,
                hops_taken,
                retrieval_failed_last=(not retrieval_succeeded_this_hop)
            )
            reasoning_trace.append(f"Hop {hops_taken}: Reasoning result -> Action='{action}', Value='{value[:100]}...'")

            if action == "ANSWER_COMPLETE":
                self.logger.info("Reasoning concluded: ANSWER_COMPLETE.")
                reasoning_trace.append(f"Hop {hops_taken}: Reasoning -> ANSWER_COMPLETE. Proceeding to final answer.")
                break
            elif action == "NEXT_QUERY":
                next_raw_query = value
                current_query = self._transform_query(next_raw_query)
                reasoning_trace.append(f"Hop {hops_taken}: Reasoning -> NEXT_QUERY = '{current_query}'")
                if not search_query_history or current_query != search_query_history[-1]:
                    search_query_history.append(current_query)
                if not graph_query_history or current_query != graph_query_history[-1]:
                    graph_query_history.append(current_query)
            else:
                self.logger.error(f"Reasoning resulted in '{action}': {value}. Stopping multihop.")
                reasoning_trace.append(f"Hop {hops_taken}: Reasoning -> {action}: {value}. Proceeding to final answer.")
                break

        if hops_taken >= max_hops and action != "ANSWER_COMPLETE":
            self.logger.info("Max hops reached. Proceeding to final answer generation.")
            reasoning_trace.append(f"Max hops ({max_hops}) reached.")

        reasoning_trace.append("--- Final Answer Generation ---")
        final_answer = self._generate_final_answer(original_query, accumulated_context_list)

        confidence_score = None
        evaluation_metrics = None
        final_context_str = "\n\n==== CONTEXT FROM HOP/SUMMARY SEPARATOR ====\n\n".join(accumulated_context_list)

        if not final_answer.startswith(("LLM_ERROR:", "I encountered an error", "Based on the retrieved")):
            confidence_score = assess_answer_confidence(final_answer, final_context_str, original_query, self.llm_interface, self.logger)
            reasoning_trace.append(f"Confidence Score: {confidence_score:.2f}" if confidence_score is not None else "Confidence Score: N/A")
            if request_evaluation:
                 evaluation_metrics = evaluate_response(original_query, final_answer, final_context_str, self.llm_interface, self.logger)
                 eval_summary = {k: v for k, v in evaluation_metrics.items() if 'rating' in k}
                 reasoning_trace.append(f"Evaluation Metrics: {eval_summary}")

        formatted_reasoning = format_reasoning_trace(reasoning_trace)
        
        graph_filename = None
        if generate_graph:
            # This call will now succeed because graph_query_history is guaranteed to exist
            graph_filename = generate_hop_graph(reasoning_trace, graph_query_history, self.graph_dir, self.logger)

        process_end_time = time.time()
        processing_time = process_end_time - process_start_time
        self.logger.info(f"--- CNT Query Process Finished ---")
        self.logger.info(f"Total Processing time: {processing_time:.2f} seconds")
        self.logger.info(f"Final Answer Snippet: {final_answer[:200]}...")

        cache_stats = self.llm_interface.get_cache_stats()
        debug_info = {
            "processing_time_s": round(processing_time, 2),
            "hops_taken": hops_taken,
            "queries_used": list(dict.fromkeys(search_query_history)),
            "final_context_length": len(final_context_str),
            "vector_db_type": type(self.vector_store).__name__,
            "embedding_cache_hits": cache_stats["embedding_cache_hits"],
            "llm_cache_hits": cache_stats["llm_cache_hits"],
            "graph_filename": graph_filename
        }

        user_rating = record_user_feedback.get('rating') if record_user_feedback else None
        user_comment = record_user_feedback.get('comment') if record_user_feedback else None
        # Make sure record_feedback has the correct signature in evaluation.py
        record_feedback(
            feedback_history=self.feedback_history,
            feedback_db_path=self.feedback_db_path,
            logger=self.logger,
            query=original_query,
            answer=final_answer,
            hop_count=hops_taken,
            final_context=final_context_str,
            reasoning_trace=reasoning_trace,
            search_query_history=list(dict.fromkeys(search_query_history)),
            user_rating=user_rating,
            user_comment=user_comment,
            evaluation_metrics=evaluation_metrics,
            confidence_score=confidence_score,
            debug_info=debug_info
        )

        return {
        "final_answer": final_answer,
        "retrieved_sources": final_context_sources, # Add this for correct printing in main.py
        "source_info": f"Synthesized from context (KG+VDB) over {hops_taken} hop(s).",
        "reasoning_trace": reasoning_trace,
        "formatted_reasoning": formatted_reasoning,
        "confidence_score": confidence_score,
        "evaluation_metrics": evaluation_metrics,
        "debug_info": debug_info
    }