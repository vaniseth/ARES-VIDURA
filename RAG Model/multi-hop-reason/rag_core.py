import logging
import time
import re
from typing import Dict, List, Tuple, Any, Optional
from difflib import SequenceMatcher

# Import components
import config
from llm_interface import LLMInterface
from vector_store import VectorStore
from evaluation import evaluate_response, assess_answer_confidence, record_feedback
from utils import format_reasoning_trace, generate_hop_graph

class CNTRagSystem:
    """
    Core RAG system orchestrating multi-hop reasoning for CNT data.
    """
    def __init__(self,
                 llm_interface: LLMInterface,
                 vector_store: VectorStore,
                 logger: logging.Logger,
                 feedback_db_path: str,
                 feedback_history: List[Dict[str, Any]], # Pass loaded history
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
            graph_dir: Directory to save reasoning graphs.
            max_context_tokens: Max estimated tokens for context window.
            char_to_token_ratio: Estimated chars per token.
            similarity_threshold: Threshold for context deduplication.
        """
        self.llm_interface = llm_interface
        self.vector_store = vector_store
        self.logger = logger
        self.feedback_db_path = feedback_db_path
        self.feedback_history = feedback_history # Use the passed list
        self.graph_dir = graph_dir
        self.max_context_tokens = max_context_tokens
        self.char_to_token_ratio = char_to_token_ratio
        self.similarity_threshold = similarity_threshold

        if not self.vector_store.is_ready():
            self.logger.critical("Vector store provided is not ready (loaded or built). RAG system may fail.")
            # Optionally raise an error here if a ready store is mandatory at init
            # raise RuntimeError("VectorStore must be loaded or built before initializing CNTRagSystem")

    # --- Query Processing Steps ---

    def _transform_query(self, query: str) -> str:
        """Basic query transformation (lowercase)."""
        self.logger.debug(f"Applying query transformations to: '{query}'")
        # Add more complex transformations if needed (unit normalization, acronym expansion)
        transformed_query = query.lower()
        if transformed_query != query:
            self.logger.info(f"Query transformed: '{query}' -> '{transformed_query}'")
        return transformed_query

    def _expand_query(self, query: str, num_expansions: int = 2) -> List[str]:
        """Generate query variations using the LLM."""
        if num_expansions <= 0:
            return [query]

        self.logger.info(f"Expanding query '{query[:100]}...' into {num_expansions} variations.")
        # (Keep the CNT-specific expansion prompt from the original script)
        expansion_prompt = f"""You are an expert search query generator specializing in scientific literature about Carbon Nanotubes (CNTs).
        Generate {num_expansions} alternative search queries for the following original question.
        The goal is to capture the same information need using different phrasing, keywords, and potential concepts found in CNT research papers.

        Original Question:
        "{query}"

        Instructions:
        1.  Understand the core scientific intent of the original question.
        2.  Create {num_expansions} distinct alternative queries.
        3.  Use relevant scientific terms (e.g., 'CVD synthesis', 'catalyst', 'chirality', 'Raman spectroscopy', 'SWCNT', 'MWCNT', 'functionalization', specific chemical formulas if relevant).
        4.  Focus on experimental parameters, material properties, characterization techniques, or applications mentioned in CNT research.
        5.  Return ONLY the alternative queries, one per line. Do not include numbering, explanations, or the original query.

        Alternative Queries:"""

        response_text = self.llm_interface.generate_response(expansion_prompt)

        if response_text.startswith("LLM_ERROR"):
            self.logger.warning(f"Query expansion failed: {response_text}")
            return [query]

        expanded_queries = [q.strip() for q in response_text.split('\n') if q.strip()]
        expanded_queries = expanded_queries[:num_expansions] # Limit to requested number

        # Include original (transformed) query and deduplicate
        all_queries = [query] + expanded_queries
        unique_queries = list(dict.fromkeys(all_queries))

        self.logger.info(f"Generated {len(unique_queries)} unique queries: {unique_queries}")
        return unique_queries

    def _reason_and_refine_query(self, original_query: str, accumulated_context: str, history: List[str], current_hop: int) -> Tuple[str, str]:
        """Uses the LLM to decide the next step (ANSWER_COMPLETE or NEXT_QUERY)."""
        self.logger.info(f"--- Hop {current_hop} Reasoning ---")
        context_snippet = accumulated_context.strip()[:500]
        self.logger.debug(f"Reasoning based on History: {' -> '.join(history)}")
        self.logger.debug(f"Context Snippet Provided:\n{context_snippet if context_snippet else 'None'}...\n")

        # (Keep the CNT-specific reasoning prompt from the original script)
        prompt = f"""You are a precise reasoning engine for a RAG system answering questions about Carbon Nanotube (CNT) research papers and data.
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

Instructions:
1. Analyze if the Accumulated Context *directly and completely* answers the Original Question.
2. Consider the search queries used so far.
3. Decide the single next action:
    a. If context fully answers: Output *exactly*: ANSWER_COMPLETE
    b. If context is insufficient, identify the *single most important missing piece* or *next logical sub-question*. Formulate a *concise, specific scientific search query* for this. Output *exactly* in this format: NEXT_QUERY: [Your concise query here]
    c. If retrieval errors occurred or context is irrelevant, output: ANSWER_COMPLETE

Reasoning Steps (Think Internally before deciding):
1. What is the core information needed to answer the Original Question?
2. Does the Accumulated Context contain this core information? List supporting evidence/snippets briefly.
3. Are there specific, critical gaps remaining? If so, what are they?
4. Based on the gaps, what is the most effective *next* query, or is the answer complete?

CRITICAL:
- Perform the reasoning steps above internally.
- Your final output MUST still be only ONE line: EITHER "ANSWER_COMPLETE" OR "NEXT_QUERY: [your query]".
- NO explanations.

Decision:"""
# Previous best answer
# CRITICAL:
# - ONE line response ONLY.
# - EITHER "ANSWER_COMPLETE" OR "NEXT_QUERY: [query]".
# - NO explanations.

        response_text = self.llm_interface.generate_response(prompt)

        # --- Parsing Logic (Keep from original script) ---
        action = "ERROR"; value = f"LLM response parsing failed. Raw: '{response_text}'"
        if response_text.startswith("LLM_ERROR"):
             self.logger.error(f"Reasoning failed due to LLM error: {response_text}")
             value = response_text; action = "ANSWER_COMPLETE" # Default to complete on error
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
                    action = "ANSWER_COMPLETE"; value = "" # Default on format failure
                    self.logger.warning(f"Unexpected reasoning response format: '{response_clean}'. Defaulting to ANSWER_COMPLETE.")
        return action, value

    def _format_context_from_chunks(self, chunks: List[Dict[str, Any]]) -> str:
        """Formats retrieved chunks into a string for context/prompt."""
        # (Keep logic from original script)
        if not chunks: return "No relevant context found."
        context_parts = []
        for chunk in chunks:
            metadata = chunk.get('metadata', {})
            score = chunk.get('score', 0.0)
            source_info = f"Source: Doc='{metadata.get('document_name', 'N/A')}', Loc={metadata.get('page_number', 'N/A')}, Score={score:.4f}"
            context_parts.append(f"[{source_info}]\n{chunk.get('text', '')}")
        return "\n\n---\n\n".join(context_parts)


    def _manage_context(self, accumulated_context_list: List[str], new_chunk_list: List[Dict[str, Any]]) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Manages context list (deduplication, size control via summarization)."""
        if not new_chunk_list:
            self.logger.debug("No new chunks to manage.")
            return accumulated_context_list, []

        # --- Deduplication (within new chunks) ---
        unique_new_chunks_by_text: Dict[str, Dict[str, Any]] = {}
        for chunk in new_chunk_list:
            text = chunk.get('text', '')
            if not text: continue
            if text not in unique_new_chunks_by_text or chunk.get('score', 0) > unique_new_chunks_by_text[text].get('score', 0):
                 unique_new_chunks_by_text[text] = chunk
        deduped_new_chunks = list(unique_new_chunks_by_text.values())
        if len(deduped_new_chunks) < len(new_chunk_list):
            self.logger.debug(f"Removed {len(new_chunk_list) - len(deduped_new_chunks)} exact duplicates from current hop.")

        # --- Deduplicate against existing context (using similarity) ---
        existing_texts = set()
        for ctx_str in accumulated_context_list:
             text_content = ctx_str
             match = re.search(r'\]\s*(.*)', ctx_str, re.DOTALL | re.IGNORECASE) or re.search(r'\]:\s*\n?(.*)', ctx_str, re.DOTALL | re.IGNORECASE)
             if match: text_content = match.group(1)
             existing_texts.add(text_content.strip())

        truly_new_chunks = []
        for chunk in deduped_new_chunks:
            text = chunk.get('text', '')
            is_similar = False
            for existing_text in existing_texts:
                 if text == existing_text or SequenceMatcher(None, text, existing_text).ratio() > self.similarity_threshold:
                     is_similar = True
                     self.logger.debug("Found new chunk identical or highly similar to existing context. Skipping.")
                     break
            if not is_similar:
                 truly_new_chunks.append(chunk)
                 existing_texts.add(text) # Add its text to check against future chunks in this hop

        if len(truly_new_chunks) < len(deduped_new_chunks):
            self.logger.info(f"Removed {len(deduped_new_chunks) - len(truly_new_chunks)} chunks similar to existing context.")

        if not truly_new_chunks:
            self.logger.info("No truly new, unique context chunks found in this hop.")
            return accumulated_context_list, []

        # Format and add the new unique context
        new_context_string = self._format_context_from_chunks(truly_new_chunks)
        current_context_list = accumulated_context_list + [new_context_string]

        # --- Size Management (Summarization) ---
        estimated_tokens = sum(len(ctx) for ctx in current_context_list) / self.char_to_token_ratio
        self.logger.debug(f"Context size before potential summarization: ~{estimated_tokens:.0f} tokens")

        while estimated_tokens > self.max_context_tokens and len(current_context_list) > 1:
             self.logger.info(f"Context too large ({estimated_tokens:.0f} > {self.max_context_tokens} tokens), summarizing oldest...")
             oldest_context = current_context_list.pop(0)

             if oldest_context.strip().startswith("[SUMMARIZED CONTEXT]:"):
                 self.logger.warning("Oldest context already summarized. Keeping.")
                 current_context_list.insert(0, oldest_context) # Put it back
                 break # Avoid summarizing summaries

             # (Keep the CNT-specific summarization prompt from the original script)
             summary_prompt = f"""Summarize the following context section from scientific literature or experimental data regarding Carbon Nanotubes (CNTs).
             Preserve all key scientific findings, experimental parameters (temperature, pressure, catalyst, etc.), numerical results, material properties, characterization data (e.g., Raman shifts, diameters), and specific CNT types (SWCNT, MWCNT, chirality) mentioned.
             Be concise but retain the essential scientific information.

             Context Section to Summarize:
             ---
             {oldest_context}
             ---

             Concise Scientific Summary:"""
             summarized = self.llm_interface.generate_response(summary_prompt)

             if summarized.startswith("LLM_ERROR"):
                 self.logger.error(f"Failed to summarize context: {summarized}. Keeping original.")
                 current_context_list.insert(0, oldest_context) # Put original back
                 break # Stop summarization if LLM fails
             else:
                 current_context_list.insert(0, f"[SUMMARIZED CONTEXT]:\n{summarized}")
                 self.logger.info("Summarization complete.")

             estimated_tokens = sum(len(ctx) for ctx in current_context_list) / self.char_to_token_ratio
             self.logger.info(f"Context size after summarization: ~{estimated_tokens:.0f} tokens")

        if estimated_tokens > self.max_context_tokens:
            self.logger.warning(f"Context size (~{estimated_tokens:.0f}) still exceeds max tokens ({self.max_context_tokens}) after summarization attempt.")

        return current_context_list, truly_new_chunks


    def _generate_final_answer(self, original_query: str, accumulated_context_list: List[str]) -> str:
        """Generates the final answer from the accumulated context."""
        self.logger.info("Generating final answer from accumulated context...")
        final_context_str = "\n\n==== CONTEXT FROM HOP/SUMMARY SEPARATOR ====\n\n".join(accumulated_context_list)

        if not final_context_str or final_context_str.strip() == "No relevant context found.":
            self.logger.warning("No context available for final answer generation.")
            return "Based on the retrieved scientific literature and data, I could not find the specific information required to answer your question."

        # (Keep the CNT-specific final answer prompt from the original script)
        prompt = f"""You are an expert scientific assistant knowledgeable about Carbon Nanotubes (CNTs).
Your task is to synthesize a final, comprehensive answer to the user's *Original Question* using *only* the information found within the *Accumulated Context* provided below.

Instructions:
1.  Read the *Original Question*.
2.  Read *all sections* within the Accumulated Context.
3.  Synthesize a single, coherent answer that directly addresses the *Original Question* with scientific accuracy.
4.  Base your answer *strictly* on the text present in the Accumulated Context. Do *not* add external information or interpretations.
5.  If the context definitively lacks the info needed, state clearly: "Based on the provided context, the specific information regarding [missing aspect] could not be found."
6.  Structure the answer logically. Start directly with the answer.
7.  Do NOT refer to the search process or source markers (`[Source: ...]`).

Original Question:
"{original_query}"

Accumulated Context:
--- START CONTEXT ---
{final_context_str}
--- END CONTEXT ---

Final Synthesized Answer (based ONLY on context):"""

        final_response = self.llm_interface.generate_response(prompt)

        if final_response.startswith("LLM_ERROR"):
             self.logger.error(f"Final answer generation failed: {final_response}")
             return f"I encountered an error while trying to generate the final answer ({final_response})."
        else:
             self.logger.info(f"Final answer generated (length: {len(final_response)}).")
             # Basic post-processing
             final_response = re.sub(r"^\s*Final Synthesized Answer:?\s*", "", final_response, flags=re.IGNORECASE).strip()
             return final_response

    # --- Main Process Method ---
    def process_query(self, question: str, top_k: int = config.DEFAULT_TOP_K, max_hops: int = config.DEFAULT_MAX_HOPS,
                      use_query_expansion: bool = False,
                      request_evaluation: bool = True,
                      generate_graph: bool = True,
                      record_user_feedback: Optional[Dict[str, Any]] = None # For explicit user feedback
                      ) -> Dict[str, Any]:
        """Processes a query using the multi-hop RAG pipeline."""
        process_start_time = time.time()
        self.logger.info(f"\n--- Starting CNT Query Process ---")
        self.logger.info(f"Original Question: '{question}'")
        self.logger.info(f"Config: top_k={top_k}, max_hops={max_hops}, expansion={use_query_expansion}, eval={request_evaluation}, graph={generate_graph}")

        if not self.vector_store.is_ready():
             self.logger.critical("Vector database is not ready. Cannot process query.")
             return {"final_answer": "Error: Knowledge base not available.", "source_info": "Initialization Error", "reasoning_trace": ["Vector DB not ready"], "formatted_reasoning": "Vector DB not ready", "confidence_score": None, "evaluation_metrics": None, "debug_info": {"processing_time_s": time.time() - process_start_time, "hops_taken": 0}}

        original_query = question
        current_query = self._transform_query(original_query)

        accumulated_context_list = []
        reasoning_trace = ["START"]
        search_query_history = [current_query] # Start with initial transformed query
        hops_taken = 0
        graph_query_history = [current_query] # History specifically for graph nodes

        # --- Initial Query Expansion ---
        if use_query_expansion:
             self.logger.info("Applying initial query expansion...")
             expanded_queries = self._expand_query(current_query, num_expansions=2)
             if expanded_queries and len(expanded_queries) > 1: # Use original + expansions
                 search_query_history = expanded_queries # Use all for potential retrieval/reasoning input? Let's just use the first for now.
                 current_query = expanded_queries[0] # Use the first (usually original transformed)
                 graph_query_history = expanded_queries # Show expansion on graph
                 reasoning_trace.append(f"Initial Expanded Queries: {expanded_queries}")
             # If expansion fails or only returns original, current_query remains the transformed original

        # --- Multihop Loop ---
        for hop in range(max_hops):
            hops_taken = hop + 1
            self.logger.info(f"\n--- Hop {hops_taken}/{max_hops} ---")
            reasoning_trace.append(f"--- Hop {hops_taken} ---")

            if not current_query:
                 self.logger.warning(f"Hop {hops_taken}: Query is empty. Breaking.")
                 reasoning_trace.append(f"Hop {hops_taken}: Error - Query empty.")
                 break

            # Retrieve context for the current query
            self.logger.info(f"Retrieving context for query: '{current_query}'")
            reasoning_trace.append(f"Hop {hops_taken}: Retrieving with query -> '{current_query}'")
            query_embedding = self.llm_interface.get_embedding(current_query, task_type="RETRIEVAL_QUERY")

            if query_embedding is None:
                self.logger.error("Query embedding failed. Cannot retrieve. Breaking hop.")
                reasoning_trace.append(f"Hop {hops_taken}: Query embedding failed.")
                # Decide whether to break entirely or try to reason with existing context
                # Let's break the hop and proceed to reasoning with what we have.
                action, value = "ANSWER_COMPLETE", "Embedding failed" # Force completion
            else:
                retrieved_chunks = self.vector_store.query(query_embedding, top_k=top_k)

                if not retrieved_chunks:
                     self.logger.warning(f"Retrieval returned no relevant chunks for hop {hops_taken}.")
                     reasoning_trace.append(f"Hop {hops_taken}: Retrieval yielded no results.")
                else:
                     self.logger.info(f"Retrieved {len(retrieved_chunks)} chunks for hop {hops_taken}.")
                     reasoning_trace.append(f"Hop {hops_taken}: Retrieved {len(retrieved_chunks)} chunk(s). Top score: {retrieved_chunks[0]['score']:.4f}")

                # Context Management
                accumulated_context_list, added_chunks = self._manage_context(accumulated_context_list, retrieved_chunks)
                if added_chunks:
                    reasoning_trace.append(f"Hop {hops_taken}: Added {len(added_chunks)} unique chunks to context.")
                elif retrieved_chunks:
                    reasoning_trace.append(f"Hop {hops_taken}: No new unique chunks added (duplicates/similar).")

                # Reasoning
                full_context_for_reasoning = "\n\n==== CONTEXT FROM HOP/SUMMARY SEPARATOR ====\n\n".join(accumulated_context_list)
                action, value = self._reason_and_refine_query(original_query, full_context_for_reasoning, graph_query_history, hops_taken)
                reasoning_trace.append(f"Hop {hops_taken}: Reasoning result -> Action='{action}', Value='{value[:100]}...'")

            # Process Reasoning Outcome
            if action == "ANSWER_COMPLETE":
                 self.logger.info("Reasoning concluded: ANSWER_COMPLETE.")
                 reasoning_trace.append(f"Hop {hops_taken}: Reasoning -> ANSWER_COMPLETE. Proceeding to final answer.")
                 break # Exit loop
            elif action == "NEXT_QUERY":
                 next_raw_query = value
                 current_query = self._transform_query(next_raw_query) # Transform the LLM generated query
                 reasoning_trace.append(f"Hop {hops_taken}: Reasoning -> NEXT_QUERY = '{current_query}'")
                 search_query_history.append(current_query) # Add refined query to overall history
                 graph_query_history.append(current_query) # Add refined query for graph nodes
                 # Loop continues with the new current_query
            else: # Includes ERROR or unexpected actions
                 self.logger.error(f"Reasoning resulted in '{action}': {value}. Stopping multihop.")
                 reasoning_trace.append(f"Hop {hops_taken}: Reasoning -> {action}: {value}. Proceeding to final answer.")
                 break # Exit loop

        # --- End of Loop ---
        if hops_taken == max_hops and action != "ANSWER_COMPLETE":
            self.logger.info("Max hops reached. Proceeding to final answer.")
            reasoning_trace.append(f"Max hops ({max_hops}) reached.")

        reasoning_trace.append("--- Final Answer Generation ---")

        # --- Final Answer Generation ---
        final_answer = self._generate_final_answer(original_query, accumulated_context_list)

        # --- Confidence & Evaluation ---
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

        # --- Format Trace & Generate Graph ---
        formatted_reasoning = format_reasoning_trace(reasoning_trace)
        graph_filename = None
        if generate_graph:
            # Use unique queries shown in graph nodes
            graph_filename = generate_hop_graph(reasoning_trace, graph_query_history, self.graph_dir, self.logger)

        # --- Wrap Up ---
        process_end_time = time.time()
        processing_time = process_end_time - process_start_time
        self.logger.info(f"--- CNT Query Process Finished ---")
        self.logger.info(f"Total Processing time: {processing_time:.2f} seconds")
        self.logger.info(f"Final Answer Snippet: {final_answer[:200]}...")

        cache_stats = self.llm_interface.get_cache_stats()
        debug_info = {
            "processing_time_s": round(processing_time, 2),
            "hops_taken": hops_taken,
            "queries_used": list(dict.fromkeys(search_query_history)), # Unique queries attempted
            "final_context_length": len(final_context_str),
            "vector_db_type": type(self.vector_store).__name__,
            "embedding_cache_hits": cache_stats["embedding_cache_hits"],
            "llm_cache_hits": cache_stats["llm_cache_hits"],
            "graph_filename": graph_filename
        }

        # --- Record Feedback (including automated metrics) ---
        user_rating = record_user_feedback.get('rating') if record_user_feedback else None
        user_comment = record_user_feedback.get('comment') if record_user_feedback else None
        record_feedback(
            feedback_history=self.feedback_history, # Pass the list to append to
            feedback_db_path=self.feedback_db_path,
            logger=self.logger,
            query=original_query,
            answer=final_answer,
            hop_count=hops_taken,
            final_context=final_context_str, # Consider truncating this?
            reasoning_trace=reasoning_trace,
            search_query_history=list(dict.fromkeys(search_query_history)),
            user_rating=user_rating,
            user_comment=user_comment,
            evaluation_metrics=evaluation_metrics,
            confidence_score=confidence_score,
            debug_info=debug_info # Include debug info in feedback log
        )

        return {
            "final_answer": final_answer,
            "source_info": f"Synthesized from context ({type(self.vector_store).__name__}) over {hops_taken} hop(s).",
            "reasoning_trace": reasoning_trace, # Raw trace
            "formatted_reasoning": formatted_reasoning,
            "confidence_score": confidence_score,
            "evaluation_metrics": evaluation_metrics,
            "debug_info": debug_info
        }