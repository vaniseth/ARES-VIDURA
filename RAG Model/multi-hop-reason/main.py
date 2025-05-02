import os
import time
import logging
from typing import List

# Import configuration and components
import config
from utils import setup_logging
from llm_interface import LLMInterface
from vector_store import get_vector_store, VectorStore
from evaluation import load_feedback_history
from rag_core import CNTRagSystem

def load_test_questions(filepath: str) -> List[str]:
    """Loads questions from a text file, one question per line."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            questions = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        return questions
    except FileNotFoundError:
        print(f"Error: Test questions file not found at {filepath}")
        return []
    except Exception as e:
        print(f"Error reading test questions file: {e}")
        return []

def run_test_query(rag_system: CNTRagSystem, question: str, index: int):
    """Runs a single test query and prints the results."""
    print(f"\n--- Processing Query {index+1} ---")
    print(f"Question: {question}")

    start_time = time.time()
    results = rag_system.process_query(
        question=question,
        top_k=config.DEFAULT_TOP_K,             # Use defaults from config
        max_hops=config.DEFAULT_MAX_HOPS,       # Use defaults from config
        use_query_expansion=True,               # Example: enable expansion
        request_evaluation=True,                # Example: enable evaluation
        generate_graph=True                     # Example: enable graph generation
    )
    end_time = time.time()

    print("\n--- Final Answer ---")
    print(results.get("final_answer", "N/A"))

    confidence = results.get("confidence_score")
    print("\n--- Confidence Score ---")
    print(f"{confidence:.2f}" if confidence is not None else "N/A")

    # print("\n--- Evaluation Metrics ---")
    # print(results.get("evaluation_metrics", "N/A"))

    # print("\n--- Formatted Reasoning Trace ---")
    # print(results.get("formatted_reasoning", "N/A"))

    print("\n--- Debug Info ---")
    debug_info = results.get("debug_info", {})
    print(f"  Processing Time: {debug_info.get('processing_time_s', 'N/A')}s")
    print(f"  Hops Taken: {debug_info.get('hops_taken', 'N/A')}")
    print(f"  Vector Store: {debug_info.get('vector_db_type', 'N/A')}")
    print(f"  Queries Used: {debug_info.get('queries_used', 'N/A')}")
    graph_file = debug_info.get("graph_filename")
    if graph_file:
        print(f"  Reasoning graph saved to: {graph_file}")
    print("-" * 30)


if __name__ == "__main__":
    # --- Setup ---
    logger = setup_logging(config.DEFAULT_LOG_LEVEL, config.DEFAULT_LOG_FILE_PATH)
    logger.info("--- Starting CNT RAG Application ---")

    # --- Initialize Components ---
    try:
        llm_interface = LLMInterface(
            api_key=config.API_KEY,
            model_id=config.DEFAULT_MODEL_ID,
            embedding_model_id=config.DEFAULT_TEXT_EMBEDDING_MODEL,
            generation_config=config.DEFAULT_GENERATION_CONFIG,
            use_embedding_cache=config.DEFAULT_USE_EMBEDDING_CACHE,
            use_llm_cache=config.DEFAULT_USE_LLM_CACHE,
            logger=logger
        )

        vector_store = get_vector_store(
            vector_db_type=config.DEFAULT_VECTOR_DB_TYPE,
            vector_db_path=config.DEFAULT_VECTOR_DB_PATH, # Path used for CSV, or persist dir for Chroma, or cache load/save for InMemory
            logger=logger
        )

        # Define chunk settings dictionary
        chunk_settings = {
            'strategy': config.DEFAULT_CHUNK_STRATEGY,
            'size': config.DEFAULT_CHUNK_SIZE,
            'overlap': config.DEFAULT_CHUNK_OVERLAP,
        }

        # Load or build the vector store *before* initializing RAG system
        logger.info("Loading or building vector store...")
        build_success = vector_store.load_or_build(
            documents_path_pattern=config.DEFAULT_DOCUMENTS_PATH_PATTERN,
            chunk_settings=chunk_settings,
            embedding_interface=llm_interface # Pass LLM interface for embedding during build
        )

        if not build_success or not vector_store.is_ready():
             logger.critical("Vector store initialization failed. Exiting.")
             exit(1)
        logger.info("Vector store ready.")

        # Load feedback history
        feedback_history = load_feedback_history(config.DEFAULT_FEEDBACK_DB_PATH, logger)

        # Initialize the main RAG system
        cnt_rag_system = CNTRagSystem(
            llm_interface=llm_interface,
            vector_store=vector_store,
            logger=logger,
            feedback_db_path=config.DEFAULT_FEEDBACK_DB_PATH,
            feedback_history=feedback_history, # Pass the loaded list
            graph_dir=config.DEFAULT_GRAPH_DIR,
            max_context_tokens=config.MAX_CONTEXT_TOKENS,
            char_to_token_ratio=config.CHAR_TO_TOKEN_RATIO,
            similarity_threshold=config.SIMILARITY_THRESHOLD
        )
        logger.info("CNTRagSystem Initialized.")

    except Exception as e:
        logger.exception(f"Fatal error during RAG system initialization: {e}")
        exit(1)

    # --- Load Test Questions ---
    test_questions = load_test_questions(config.DEFAULT_TEST_QUESTIONS_PATH)

    # --- Run Test Queries ---
    if not test_questions:
        logger.warning("No test questions loaded. Exiting.")
    else:
        logger.info(f"Loaded {len(test_questions)} test questions. Starting processing...")
        overall_start_time = time.time()
        for i, question in enumerate(test_questions):
            try:
                run_test_query(cnt_rag_system, question, i)
            except Exception as query_err:
                 logger.exception(f"Error processing query {i+1} ('{question[:50]}...'): {query_err}")
                 print(f"\n--- Error processing query {i+1}. Check logs. ---")
        overall_end_time = time.time()
        logger.info(f"Finished processing all {len(test_questions)} questions in {overall_end_time - overall_start_time:.2f} seconds.")

    logger.info("--- CNT RAG Application Finished ---")
