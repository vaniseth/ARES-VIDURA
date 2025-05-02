import google.generativeai as genai
import logging
import time
import re
from tenacity import retry, stop_after_attempt, wait_random_exponential
from typing import Dict, List, Optional, Any

import config # Import config for defaults

class LLMInterface:
    """Handles interactions with Google Generative AI models for embeddings and generation."""

    def __init__(self,
                 api_key: str,
                 model_id: str = config.DEFAULT_MODEL_ID,
                 embedding_model_id: str = config.DEFAULT_TEXT_EMBEDDING_MODEL,
                 generation_config: Optional[Dict] = None,
                 use_embedding_cache: bool = config.DEFAULT_USE_EMBEDDING_CACHE,
                 use_llm_cache: bool = config.DEFAULT_USE_LLM_CACHE,
                 logger: logging.Logger = logging.getLogger("CNTRAG")):
        """
        Initializes the LLM Interface.

        Args:
            api_key: Google API Key.
            model_id: Generative model ID.
            embedding_model_id: Embedding model ID.
            generation_config: Configuration for the generative model.
            use_embedding_cache: Enable in-memory embedding cache.
            use_llm_cache: Enable in-memory LLM response cache.
            logger: Logger instance.
        """
        self.logger = logger
        self.api_key = api_key
        self.model_id = model_id
        self.embedding_model_id = embedding_model_id
        self.embedding_model_name = f"models/{embedding_model_id}"
        self.generation_config = generation_config or config.DEFAULT_GENERATION_CONFIG

        self.use_embedding_cache = use_embedding_cache
        self.embedding_cache: Dict[str, List[float]] = {}
        self.use_llm_cache = use_llm_cache
        self.llm_cache: Dict[str, str] = {}
        # Track cache hits (simple example)
        self.embedding_cache_hits = 0
        self.llm_cache_hits = 0

        try:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(
                model_name=self.model_id,
                generation_config=self.generation_config,
                # safety_settings={'HARASSMENT':'BLOCK_NONE', ...} # Add if needed
            )
            self.logger.info(f"Initialized Generative Model: {self.model_id}")
            self.logger.info(f"Initialized Embedding Model: {self.embedding_model_id}")
        except Exception as e:
            self.logger.error(f"Error initializing Google AI clients: {e}")
            raise

    @retry(wait=wait_random_exponential(multiplier=1, min=2, max=60), stop=stop_after_attempt(5))
    def get_embedding(self, text: str, task_type="RETRIEVAL_DOCUMENT") -> Optional[List[float]]:
        """Generates embeddings with retry logic and caching."""
        cache_key = f"{task_type}::{text}"
        if self.use_embedding_cache and cache_key in self.embedding_cache:
             self.logger.debug(f"Embedding cache HIT for task '{task_type}'")
             self.embedding_cache_hits += 1
             return self.embedding_cache[cache_key]

        if not text or not text.strip():
            self.logger.warning(f"Attempted to embed empty text for task '{task_type}'. Returning None.")
            return None

        self.logger.debug(f"Getting embedding for task '{task_type}' (text length: {len(text)})...")
        try:
            response = genai.embed_content(
                model=self.embedding_model_name,
                content=text,
                task_type=task_type,
            )
            embedding = response.get('embedding')
            if embedding and isinstance(embedding, list) and len(embedding) > 0:
                if self.use_embedding_cache:
                    self.embedding_cache[cache_key] = embedding
                return embedding
            else:
                self.logger.error(f"Embedding API returned invalid/empty embedding for task '{task_type}'. Response: {response}")
                return None
        except Exception as e:
            error_str = str(e).lower()
            # (Keep existing retry/error handling logic from the original script)
            if "resource has been exhausted" in error_str or "quota" in error_str or "429" in error_str:
                self.logger.warning(f"Rate limit or quota error during embedding: {e}. Retrying...")
                raise # Re-raise to trigger tenacity retry
            elif "api key not valid" in error_str or "permission denied" in error_str or "401" in error_str or "403" in error_str:
                self.logger.error(f"Authentication/Permission Error during embedding: {e}. NOT RETRYING.")
                return None
            elif "500" in error_str or "internal server error" in error_str:
                self.logger.warning(f"Server error during embedding: {e}. Retrying...")
                raise
            elif "invalid content" in error_str or "400" in error_str:
                 log_snippet = text[:100].encode('unicode_escape').decode('utf-8')
                 self.logger.error(f"Invalid content error during embedding: {e}. Text snippet (escaped): '{log_snippet}...'. NOT RETRYING.")
                 return None
            else:
                self.logger.error(f"Non-retryable or unexpected error generating embeddings: {e}")
                return None

    def batch_get_embeddings(self, text_list: List[str], batch_size=10, task_type="RETRIEVAL_DOCUMENT") -> List[Optional[List[float]]]:
        """Process embeddings for a list of texts sequentially with retries."""
        self.logger.info(f"Getting embeddings for {len(text_list)} texts (sequential calls)")
        all_embeddings: List[Optional[List[float]]] = []
        rate_limit_delay = 0.5

        for i, text in enumerate(text_list):
            self.logger.debug(f"Processing embedding for item {i+1}/{len(text_list)}")
            emb = None
            try:
                emb = self.get_embedding(text, task_type=task_type) # Use the method with retry
                if emb is None:
                    self.logger.warning(f"Embedding failed for item {i+1} after retries.")
            except Exception as e:
                self.logger.error(f"Embedding failed permanently for item {i+1} after retries: {e}")

            all_embeddings.append(emb)

            # Simplified delay - consider more sophisticated rate limiting if needed
            if (i + 1) % batch_size == 0:
                try:
                    time.sleep(rate_limit_delay)
                except KeyboardInterrupt:
                    self.logger.warning("Embedding process interrupted.")
                    break

            if (i + 1) % 100 == 0 or (i + 1) == len(text_list):
                successful_count = sum(1 for e in all_embeddings if e is not None)
                self.logger.info(f"Processed {i + 1}/{len(text_list)} embeddings ({successful_count} successful).")

        return all_embeddings


    def generate_response(self, prompt: str) -> str:
        """Generates a response from the LLM with caching and error handling."""
        if self.use_llm_cache and prompt in self.llm_cache:
             self.logger.debug("LLM cache HIT")
             self.llm_cache_hits += 1
             return self.llm_cache[prompt]

        self.logger.debug(f"Generating LLM response for prompt (length: {len(prompt)}): '{prompt[:200]}...'")
        start_time = time.time()

        try:
            response = self.model.generate_content(prompt)

            # (Keep existing response parsing/error handling logic from the original script)
            if not response.candidates:
                feedback = getattr(response, 'prompt_feedback', None)
                block_reason = "Unknown"
                if feedback: block_reason = getattr(feedback, 'block_reason', 'Unknown')
                self.logger.warning(f"LLM Response blocked or had no candidates. Reason: {block_reason}.")
                error_message = f"LLM_ERROR: Response blocked (Reason: {block_reason})."
                if self.use_llm_cache: self.llm_cache[prompt] = error_message
                return error_message

            generated_text = ""
            if hasattr(response, 'text') and response.text:
                 generated_text = response.text
            else:
                try:
                     if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                         generated_text = "".join(part.text for part in response.candidates[0].content.parts if hasattr(part, 'text'))
                         if not generated_text:
                             self.logger.warning("LLM Response generated but text content is empty even in parts.")
                             generated_text = "LLM_ERROR: Response content was empty."
                     else:
                         self.logger.warning("LLM Response generated but text attribute and parts are missing/empty.")
                         generated_text = "LLM_ERROR: Response structure unexpected or empty."
                except (IndexError, AttributeError, Exception) as e:
                     self.logger.error(f"Error accessing response parts: {e}")
                     generated_text = "LLM_ERROR: Failed to access response content."

            elapsed_time = time.time() - start_time
            self.logger.debug(f"LLM generation took {elapsed_time:.2f}s. Response length: {len(generated_text)}")

            if self.use_llm_cache:
                 self.llm_cache[prompt] = generated_text

            return generated_text

        except Exception as e:
            self.logger.exception(f"Exception during LLM generation call: {e}")
            error_str = str(e).lower()
            if "resource_exhausted" in error_str or "quota" in error_str or "429" in error_str:
                return "LLM_ERROR: API rate limit or quota exceeded."
            elif "api key not valid" in error_str or "permission denied" in error_str or "401" in error_str or "403" in error_str:
                return "LLM_ERROR: Invalid API Key or insufficient permissions."
            return f"LLM_ERROR: An unexpected error occurred during text generation: {str(e)}"

    def get_cache_stats(self) -> Dict[str, int]:
        """Returns the current cache hit counts."""
        return {
            "embedding_cache_hits": self.embedding_cache_hits,
            "llm_cache_hits": self.llm_cache_hits
        }
