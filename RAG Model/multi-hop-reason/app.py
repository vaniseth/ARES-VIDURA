# app.py
import streamlit as st
from typing import List, Dict, Any

# Import your existing RAG components
import config
from utils import setup_logging
from llm_interface import LLMInterface
from vector_store import get_vector_store
from rag_core import CNTRagSystem
from evaluation import load_feedback_history

# --- App Configuration & Title ---
st.set_page_config(page_title="CNT Research Assistant", layout="wide")
st.title("🔬 CNT Research Assistant")
st.info("Ask a question about Carbon Nanotubes based on the provided research papers. The chatbot remembers the conversation history.")

# --- Initialization & Caching ---
# Use Streamlit's caching to load the RAG system only once.
@st.cache_resource
def load_rag_system():
    """Loads all necessary components and initializes the CNTRagSystem."""
    logger = setup_logging(config.DEFAULT_LOG_LEVEL, "logs/streamlit_app.log")
    logger.info("--- Initializing Streamlit App and RAG System ---")

    try:
        llm_interface = LLMInterface(
            llm_provider=config.DEFAULT_GENERATIVE_LLM_PROVIDER,
            google_api_key=config.GOOGLE_API_KEY,
            openai_api_key=config.OPENAI_API_KEY,
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
             logger.critical("Vector store initialization failed. App may not function.")
             st.error("Fatal Error: The knowledge base (Vector Store) could not be loaded. Please check the logs.")
             return None
        logger.info("Vector store ready.")

        feedback_history = load_feedback_history(config.DEFAULT_FEEDBACK_DB_PATH, logger)

        # Assuming user_type is 'advanced' for the UI, or you can add a selector
        rag_system = CNTRagSystem(
            llm_interface=llm_interface,
            vector_store=vector_store,
            logger=logger,
            feedback_db_path=config.DEFAULT_FEEDBACK_DB_PATH,
            feedback_history=feedback_history,
            user_type="advanced" # Or make this selectable in the UI
        )
        logger.info("--- CNTRagSystem Initialized Successfully for Streamlit ---")
        return rag_system

    except Exception as e:
        st.error(f"An error occurred during initialization: {e}")
        st.exception(e)
        return None

# Load the system
rag_system = load_rag_system()

# --- Chat History Management ---
# Initialize chat history in Streamlit's session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display past messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # If the message is from the assistant and has sources, display them
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("View Sources"):
                for source in message["sources"]:
                    doc_name = source.get('document_name', 'N/A')
                    page_num = source.get('page_number', 'N/A')
                    st.info(f"**Document:** {doc_name}\n\n**Page:** {page_num}")


# --- Main Chat Logic ---
if rag_system:
    # Get user input from the chat box
    if prompt := st.chat_input("Ask your question about CNTs..."):
        # Add user message to history and display it
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Use the new method to generate a context-aware query
                contextual_query = rag_system.generate_contextual_query(
                    chat_history=st.session_state.messages,
                    new_question=prompt
                )
                st.info(f"**Standalone Query:** *{contextual_query}*") # Display the generated query for transparency

                # Process the query using the RAG system
                response_data = rag_system.process_query(
                    question=contextual_query,
                    top_k=config.DEFAULT_TOP_K,
                    max_hops=config.DEFAULT_MAX_HOPS,
                    use_query_expansion=True,
                    request_evaluation=True,
                    generate_graph=False # Disable graph generation for faster UI response
                )

                final_answer = response_data["final_answer"]
                retrieved_sources = response_data.get("retrieved_sources", [])

                # Display the final answer
                st.markdown(final_answer)

                # Display sources in an expander
                if retrieved_sources and isinstance(retrieved_sources, list):
                    with st.expander("View Sources"):
                        for source in retrieved_sources:
                            doc_name = source.get('document_name', 'N/A')
                            page_num = source.get('page_number', 'N/A')
                            st.info(f"**Document:** {doc_name}\n\n**Page:** {page_num}")

                # Add the complete assistant response to history
                assistant_message = {
                    "role": "assistant",
                    "content": final_answer,
                    "sources": retrieved_sources if isinstance(retrieved_sources, list) else []
                }
                st.session_state.messages.append(assistant_message)
else:
    st.warning("RAG system could not be initialized. The chatbot is offline.")