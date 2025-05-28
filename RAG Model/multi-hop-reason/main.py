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
    print(f"\n--- Processing Query {index+1} for '{rag_system.user_type}' user ---") # Added user type to print
    print(f"Question: {question}")

    start_time = time.time()
    results = rag_system.process_query(
        question=question,
        top_k=config.DEFAULT_TOP_K,            
        max_hops=config.DEFAULT_MAX_HOPS,      
        use_query_expansion=True,              
        request_evaluation=True,               
        generate_graph=True                    
    )
    end_time = time.time()

    print("\n--- Final Answer ---")
    print(results.get("final_answer", "N/A"))

    confidence = results.get("confidence_score")
    print("\n--- Confidence Score ---")
    print(f"{confidence:.2f}" if confidence is not None else "N/A")

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

    # --- Determine User Type (Example: via input) ---
    user_type_input_str = 'advanced' #input("Enter user type ('novice' or 'advanced', press Enter for 'novice'): ").strip().lower()
    if not user_type_input_str:
        user_type_input_str = "advanced"
    elif user_type_input_str not in ["novice", "advanced"]:
        logger.warning(f"Invalid user type '{user_type_input_str}' entered. Defaulting to 'novice'.")
        user_type_input_str = "advanced"
    logger.info(f"Selected user type: {user_type_input_str}")


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
            vector_db_path=config.DEFAULT_VECTOR_DB_PATH, 
            logger=logger
        )

        chunk_settings = {
            'strategy': config.DEFAULT_CHUNK_STRATEGY,
            'size': config.DEFAULT_CHUNK_SIZE,
            'overlap': config.DEFAULT_CHUNK_OVERLAP,
        }

        logger.info("Loading or building vector store...")
        build_success = vector_store.load_or_build(
            documents_path_pattern=config.DEFAULT_DOCUMENTS_PATH_PATTERN,
            chunk_settings=chunk_settings,
            embedding_interface=llm_interface 
        )

        if not build_success or not vector_store.is_ready():
             logger.critical("Vector store initialization failed. Exiting.")
             exit(1)
        logger.info("Vector store ready.")

        feedback_history = load_feedback_history(config.DEFAULT_FEEDBACK_DB_PATH, logger)

        cnt_rag_system = CNTRagSystem(
            llm_interface=llm_interface,
            vector_store=vector_store,
            logger=logger,
            feedback_db_path=config.DEFAULT_FEEDBACK_DB_PATH,
            feedback_history=feedback_history, 
            user_type=user_type_input_str, # Pass the user type here
            graph_dir=config.DEFAULT_GRAPH_DIR,
            max_context_tokens=config.MAX_CONTEXT_TOKENS,
            char_to_token_ratio=config.CHAR_TO_TOKEN_RATIO,
            similarity_threshold=config.SIMILARITY_THRESHOLD
        )
        logger.info(f"CNTRagSystem Initialized for '{cnt_rag_system.user_type}' user.")

    except Exception as e:
        logger.exception(f"Fatal error during RAG system initialization: {e}")
        exit(1)

    test_questions = load_test_questions(config.DEFAULT_TEST_QUESTIONS_PATH)

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