# app.py (Final Polished Version)
import streamlit as st
import os
import re # Import re for the utility function
import textwrap # Import textwrap for the utility function
from typing import List, Dict, Any

from streamlit_agraph import agraph, Node, Edge, Config

import config
from utils import setup_logging
from llm_interface import LLMInterface
from vector_store import get_vector_store
from graph_db import Neo4jGraphDB
from rag_core import CNTRagSystem
from evaluation import load_feedback_history

# --- App Configuration & Title ---
st.set_page_config(page_title="CNT Research Assistant", layout="wide")
st.title("🔬 CNT Research Assistant")
st.info("Ask a question about Carbon Nanotubes. Expand the 'Details' section under each answer to verify sources and reasoning.")

# --- Utility Function ---
# (Moved prepare_interactive_graph_data here to make app.py self-contained for UI logic)
def prepare_interactive_graph_data(reasoning_trace: List[str], query_history: List[str]) -> dict:
    nodes, edges = [], []
    def wrap_text(text, width=30): return '\n'.join(textwrap.wrap(text, width=width))
    start_query = query_history[0] if query_history else "Initial Query"
    nodes.append(Node(id='start', label=f"Start\n{wrap_text(start_query)}", shape='diamond', color='#ADD8E6'))
    last_node_id = 'start'
    hop_num, query_idx = 0, 0
    processed_reasoning_nodes = set()
    for step in reasoning_trace:
        if step.startswith("--- Hop"):
            hop_num = int(re.search(r'\d+', step).group())
            query_text = query_history[query_idx] if query_idx < len(query_history) else f"Query for Hop {hop_num}"
            query_node_id = f"query_{hop_num}"
            nodes.append(Node(id=query_node_id, label=f"Hop {hop_num} Query\n{wrap_text(query_text)}", color='#90EE90'))
            edges.append(Edge(source=last_node_id, target=query_node_id, label=f"Refine (Hop {hop_num})"))
            last_node_id = query_node_id
            if any(f"Hop {hop_num}: Reasoning -> NEXT_QUERY" in s for s in reasoning_trace if s.startswith(f"Hop {hop_num}")):
                query_idx = min(query_idx + 1, len(query_history) - 1)
        elif "Reasoning result" in step:
            reasoning_node_id = f"reasoning_{hop_num}"
            if reasoning_node_id not in processed_reasoning_nodes:
                action_match = re.search(r"Action='([^']*)'", step)
                value_match = re.search(r"Value='([^']*)", step)
                action = action_match.group(1) if action_match else "N/A"
                value = value_match.group(1).replace("'...", "") if value_match else "N/A"
                nodes.append(Node(id=reasoning_node_id, label=f"Hop {hop_num} Reasoning\nAction: {action}\nValue: {wrap_text(value)}", color='#FFFFE0', shape='ellipse'))
                edges.append(Edge(source=last_node_id, target=reasoning_node_id, label="Process Context"))
                last_node_id = reasoning_node_id
                processed_reasoning_nodes.add(reasoning_node_id)
    nodes.append(Node(id='final_answer', label='Final Answer Generation', shape='octagon', color='#FFA500'))
    edges.append(Edge(source=last_node_id, target='final_answer', label="Proceed to Final"))
    return {"nodes": nodes, "edges": edges}

# --- Initialization & Caching ---
@st.cache_resource
def load_rag_system():
    # ... (This entire function remains the same as your working version)
    log_dir = os.path.dirname(config.DEFAULT_LOG_FILE_PATH)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logger = setup_logging(config.DEFAULT_LOG_LEVEL, config.DEFAULT_LOG_FILE_PATH)
    logger.info("--- Initializing Streamlit App and RAG System ---")
    try:
        llm_interface = LLMInterface(llm_provider=config.DEFAULT_GENERATIVE_LLM_PROVIDER, google_api_key=config.GOOGLE_API_KEY, openai_api_key=config.OPENAI_API_KEY, logger=logger)
        graph_db = Neo4jGraphDB(logger=logger)
        vector_store = get_vector_store(vector_db_type=config.DEFAULT_VECTOR_DB_TYPE, vector_db_path=config.DEFAULT_VECTOR_DB_PATH, logger=logger)
        chunk_settings = {'size': config.DEFAULT_CHUNK_SIZE, 'overlap': config.DEFAULT_CHUNK_OVERLAP}
        build_success = vector_store.load_or_build(documents_path_pattern=config.DEFAULT_DOCUMENTS_PATH_PATTERN, chunk_settings=chunk_settings, embedding_interface=llm_interface, graph_db=graph_db)
        if not build_success or not vector_store.is_ready():
             logger.critical("Vector store initialization failed.")
             st.error("Fatal Error: The knowledge base could not be loaded. Please check the logs.")
             return None
        logger.info("Vector store and Knowledge Graph are ready.")
        feedback_history = load_feedback_history(config.DEFAULT_FEEDBACK_DB_PATH, logger)
        rag_system = CNTRagSystem(llm_interface=llm_interface, vector_store=vector_store, graph_db=graph_db, logger=logger, feedback_db_path=config.DEFAULT_FEEDBACK_DB_PATH, feedback_history=feedback_history, user_type="advanced")
        logger.info("--- CNTRagSystem Initialized Successfully for Streamlit ---")
        return rag_system
    except Exception as e:
        logger.exception(f"An error occurred during RAG system initialization: {e}")
        st.error(f"An error occurred during initialization: {e}")
        st.exception(e)
        return None

rag_system = load_rag_system()

# --- UI Helper Functions ---
def display_sources(sources, message_key):
    # ... (This function remains the same as your working version)
    if not sources:
        st.caption("No specific sources were cited for this response.")
        return
    for i, source in enumerate(sources):
        doc_name = source.get('document_name', 'N/A')
        page_num = source.get('page_number', 'N/A')
        chunk_id = source.get('chunk_id', f'unknown_{i}')
        button_key = f"btn_{message_key}_{i}"
        col1, col2 = st.columns([4, 1])
        with col1: st.info(f"**Source {i+1}:** {doc_name} (Page: {page_num})")
        with col2:
            if st.button("Show Text", key=button_key, use_container_width=True):
                st.session_state[f'show_chunk_{button_key}'] = not st.session_state.get(f'show_chunk_{button_key}', False)
        if st.session_state.get(f'show_chunk_{button_key}', False):
            with st.spinner("Fetching chunk text..."):
                chunk_data = rag_system.get_chunk_by_id(chunk_id)
                if chunk_data and 'chunk_text' in chunk_data:
                    st.text_area("Full Chunk Text", value=chunk_data['chunk_text'], height=200, disabled=True, key=f'text_{button_key}')
                else: st.error("Could not retrieve the full text for this chunk.")

@st.dialog("Interactive Reasoning Flow")
def show_graph_dialog(graph_data):
    st.info("You can drag nodes to rearrange the graph and use your mouse wheel to zoom.")
    if graph_data:
        config = Config(width=800, height=600, directed=True, physics=False, hierarchical={"enabled": True, "sortMethod": "directed", "shakeTowards": "roots"}, node={'size': 25, 'font': {'size': 12}}, edge={'font': {'align': 'top'}})
        agraph(nodes=graph_data["nodes"], edges=graph_data["edges"], config=config)

# --- Chat History Management ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I am a specialized research assistant for Carbon Nanotubes. How can I help you today?", "sources": [], "reasoning": "Initial greeting.", "graph_data": None}]

# Display past messages
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            # --- THE FIX: Use an expander that contains the tabs ---
            with st.expander("Show Details (Sources, Reasoning, Flow)"):
                source_tab, reasoning_tab, graph_tab = st.tabs(["🔬 Sources", "🧠 Reasoning Text", "📈 Reasoning Flow"])
                with source_tab:
                    display_sources(message.get("sources", []), f"msg_{i}")
                with reasoning_tab:
                    st.text(message.get("reasoning", "No reasoning trace available."))
                with graph_tab:
                    if message.get("graph_data"):
                        if st.button("Launch Interactive Graph", key=f"graph_btn_{i}"):
                            show_graph_dialog(message.get("graph_data"))
                    else:
                        st.caption("Graph data not generated for this response.")

# --- Main Chat Logic ---
if rag_system:
    if prompt := st.chat_input("Ask your question about CNTs..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.rerun() # Rerun to immediately display the user's message

    # Process the last user message if it hasn't been processed yet
    last_message = st.session_state.messages[-1]
    if last_message["role"] == "user" and "processed" not in last_message:
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking, analyzing, and retrieving..."):
                contextual_query = rag_system.generate_contextual_query(
                    chat_history=st.session_state.messages[:-1], # Don't include the current question
                    new_question=last_message["content"]
                )
                
                response_data = rag_system.process_query(
                    question=contextual_query,
                    top_k=config.DEFAULT_TOP_K,
                    max_hops=config.DEFAULT_MAX_HOPS,
                    use_query_expansion=True,
                    request_evaluation=False,
                    generate_graph=True # IMPORTANT: This must be True
                )

                graph_data = prepare_interactive_graph_data(
                    response_data.get("reasoning_trace", []),
                    response_data.get("debug_info", {}).get("queries_used", [])
                ) if response_data.get("reasoning_trace") else None

                assistant_message = {
                    "role": "assistant",
                    "content": response_data.get("final_answer", "Sorry, an error occurred.") + response_data.get("proactive_suggestions", ""),
                    "sources": response_data.get("retrieved_sources", []),
                    "reasoning": response_data.get("formatted_reasoning", "Trace not generated."),
                    "graph_data": graph_data
                }
                st.session_state.messages.append(assistant_message)
                st.session_state.messages[-2]["processed"] = True # Mark user message as processed
                st.rerun()
else:
    st.warning("RAG system could not be initialized. The chatbot is offline.")