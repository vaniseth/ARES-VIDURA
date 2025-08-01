# data_processing.py
import os
import glob
import json
import PyPDF2
import logging
import hashlib
from typing import List, Optional, Dict, Any, Tuple
from langchain_text_splitters import RecursiveCharacterTextSplitter
import PyPDF2
import config
# Import the new graph DB class
from graph_db import Neo4jGraphDB
from llm_interface import LLMInterface
from unstructured.partition.auto import partition

logger = logging.getLogger("CNTRAG_DataProcessing")

def _create_chunk_id(doc_name: str, element_idx: int, sub_chunk_idx: int, text_content: str) -> str:
    """Creates a unique and deterministic chunk ID."""
    # Using a hash of the content makes the ID stable even if parsing order changes slightly
    hasher = hashlib.sha1()
    # Use a prefix of the content to keep it fast
    hasher.update(text_content[:256].encode('utf-8'))
    content_hash = hasher.hexdigest()[:8]
    return f"{os.path.splitext(doc_name)[0]}_elem{element_idx}_sub{sub_chunk_idx}_{content_hash}"


def _elements_to_dicts(elements: List[Any], doc_name: str, doc_path: str) -> List[Dict[str, Any]]:
    """
    Converts unstructured elements to a list of dicts with standardized keys.
    """
    extracted_data = []
    for i, element in enumerate(elements):
        element_type = type(element).__name__
        text_content = str(element)

        base_metadata = {
            "document_name": doc_name,
            "document_path": doc_path,
            "element_index_in_doc": i,
            "element_type": element_type,
        }

        # Add page number if available
        if hasattr(element, 'metadata') and hasattr(element.metadata, 'page_number'):
            base_metadata['page_number'] = element.metadata.page_number

        content_type = "text"
        if isinstance(element, Table):
            content_type = "table"
            try:
                from unstructured.cleaners.translate import translate_html_to_md
                text_content = translate_html_to_md(str(element))
                base_metadata["table_as_markdown"] = True
            except Exception:
                base_metadata["table_as_markdown"] = False

        base_metadata["content_type"] = content_type

        if text_content.strip():
            extracted_data.append({
                "raw_text": text_content,
                "metadata": base_metadata,
            })
    return extracted_data

# --- NEW FUNCTION: To extract entities using an LLM ---
def _extract_entities_from_chunk(chunk_text: str, llm_interface: LLMInterface) -> List[Dict[str, str]]:
    """
    Uses an LLM to extract key scientific entities from a text chunk.
    """
    prompt = f"""
    From the following scientific text about Carbon Nanotubes, extract key entities.
    The entity types to extract are: "Method", "Catalyst", "Substrate", "CNT_Type", "Carbon_Source".
    - "Method": e.g., 'CVD', 'Laser Ablation', 'Arc Discharge', 'Plasma Enhanced CVD', 'Fixed Catalyst', 'Floating Catalyst'
    - "Catalyst": e.g., 'Iron', 'Nickel', 'Fe', 'Ni', 'Gadolinium'
    - "Substrate": e.g., 'Silicon', 'Graphite', 'Alumina', 'Magnesium Aluminate'
    - "CNT_Type": e.g., 'SWCNT', 'MWCNT', 'Single-walled', 'Multi-walled'
    - "Carbon_Source": e.g., 'Acetylene', 'Methane', 'Ethylene', 'C2H2', 'C2H4'

    Return the results as a JSON list of objects, where each object has a "type" and "name".
    If no entities are found, return an empty list [].

    Example:
    Text: "We grew MWCNTs on a silicon substrate using a fixed iron catalyst via the CVD method with acetylene gas."
    Output:
    [
        {{"type": "CNT_Type", "name": "MWCNT"}},
        {{"type": "Substrate", "name": "Silicon"}},
        {{"type": "Catalyst", "name": "Iron"}},
        {{"type": "Method", "name": "Fixed Catalyst"}},
        {{"type": "Method", "name": "CVD"}},
        {{"type": "Carbon_Source", "name": "Acetylene"}}
    ]

    Now, analyze this text:
    --- TEXT START ---
    {chunk_text}
    --- TEXT END ---

    JSON Output:
    """
    response_str = llm_interface.generate_response(prompt)
    try:
        # Basic cleanup in case LLM adds markdown backticks
        response_str = response_str.strip().replace("```json", "").replace("```", "").strip()
        entities = json.loads(response_str)
        if isinstance(entities, list):
            # Validate structure
            validated_entities = [e for e in entities if isinstance(e, dict) and "type" in e and "name" in e]
            return validated_entities
        return []
    except (json.JSONDecodeError, TypeError):
        logger.warning(f"Failed to decode LLM response into JSON for entity extraction. Raw response: {response_str[:100]}...")
        return []

# This is a new helper function for our smart chunking
def _create_chunk_id(doc_name: str, element_idx: int, chunk_idx: int) -> str:
    """Creates a unique ID for a chunk based on its document and position."""
    # We remove the content hash for simplicity as position is now more stable
    return f"{os.path.splitext(doc_name)[0]}_elem{element_idx}_chunk{chunk_idx}"

# --- THE NEW, FULLY REVISED parse_and_chunk_documents ---
def parse_and_chunk_documents(
    documents_path_pattern: str,
    graph_db: Neo4jGraphDB,
    embedding_interface: LLMInterface,
    chunk_size: int = config.DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = config.DEFAULT_CHUNK_OVERLAP,
    logger_parent: logging.Logger = logging.getLogger("CNTRAG")
) -> List[Dict[str, Any]]:
    """
    Parses documents using `unstructured` to preserve hierarchy, creates smart chunks,
    extracts entities, and populates both a vector store and a knowledge graph.
    """
    global logger
    logger = logger_parent

    logger.info(f"Starting 'Smart Chunking' document parsing with `unstructured`...")
    document_files = glob.glob(documents_path_pattern)
    document_files = [f for f in document_files if os.path.isfile(f) and not os.path.basename(f).startswith('~')]

    if not document_files:
        logger.error(f"No document files found matching pattern: {documents_path_pattern}")
        return []

    all_final_chunks_for_vector_store: List[Dict[str, Any]] = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    for doc_path in document_files:
        doc_name = os.path.basename(doc_path)
        logger.info(f"Processing document: {doc_name} with `unstructured`")

        try:
            # `unstructured` partitions the document into semantic elements
            elements = partition(filename=doc_path, strategy="hi_res")
            
            doc_chunks_for_kg = []
            
            # --- Smart Chunking Logic ---
            for i, element in enumerate(elements):
                element_type = type(element).__name__
                # Titles and headers are important context
                if element_type == "Title" or "Header" in element_type:
                    # We merge titles with the text that follows them for better context
                    if i + 1 < len(elements):
                        # Prepend title to the text of the next element
                        next_element_text = str(elements[i+1])
                        elements[i+1].text = f"{str(element)}\n\n{next_element_text}"
                        logger.debug(f"Merged title '{str(element)}' with next element.")
                    continue # Skip processing the title as a separate chunk

                # Split larger text elements, but keep smaller ones whole
                if len(str(element).strip()) > chunk_overlap:
                    sub_chunks = text_splitter.split_text(str(element))
                else:
                    sub_chunks = [str(element)]
                
                for j, chunk_text in enumerate(sub_chunks):
                    cleaned_chunk = chunk_text.strip()
                    if len(cleaned_chunk) < config.MIN_CHUNK_LENGTH:
                        continue

                    chunk_id = _create_chunk_id(doc_name, i, j)
                    
                    # Create rich metadata for each chunk
                    chunk_metadata = {
                        "document_name": doc_name,
                        "chunk_id": chunk_id,
                        "element_type": element_type, # e.g., 'NarrativeText', 'ListItem'
                        "page_number": getattr(element.metadata, 'page_number', None)
                    }

                    chunk_info = {
                        "chunk_text": cleaned_chunk,
                        "metadata": chunk_metadata
                    }
                    doc_chunks_for_kg.append(chunk_info)
                    all_final_chunks_for_vector_store.append(chunk_info)
            
            # --- KG Population (Same as before, but with better chunks) ---
            if doc_chunks_for_kg:
                graph_db.add_document_and_chunks(doc_name, doc_chunks_for_kg)
                logger.info(f"Extracting and linking entities for {len(doc_chunks_for_kg)} smart chunks from {doc_name}...")
                for chunk in doc_chunks_for_kg:
                    entities = _extract_entities_from_chunk(chunk['chunk_text'], embedding_interface)
                    if entities:
                        graph_db.link_chunk_to_entities(chunk['metadata']['chunk_id'], entities)

        except Exception as e:
            logger.exception(f"Failed to process file {doc_path} with `unstructured`: {e}")
            
    logger.info(f"Finished parsing. Total smart chunks created for vector store: {len(all_final_chunks_for_vector_store)}")
    return all_final_chunks_for_vector_store


# Fallback basic parser (ensure it's defined or imported correctly)
def parse_document_text(doc_path: str, logger_param: logging.Logger) -> Optional[List[Tuple[int, str]]]:
    """Basic text extraction for PDF/DOCX (simplified fallback)."""
    logger_param.debug(f"Basic parse_document_text called for {doc_path}")
    content_list = []
    try:
        if doc_path.lower().endswith(".pdf"):
            with open(doc_path, "rb") as file:
                pdf_reader = PyPDF2.PdfReader(file)
                if pdf_reader.is_encrypted:
                    logger_param.warning(f"PDF {doc_path} is encrypted. Skipping basic parsing.")
                    return None
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    if page_text and page_text.strip():
                        content_list.append((page_num + 1, page_text))
        elif doc_path.lower().endswith((".docx")):
            import docx
            document = docx.Document(doc_path)
            full_text_parts = [p.text for p in document.paragraphs if p.text.strip()]
            if full_text_parts:
                content_list.append((1, "\n".join(full_text_parts))) # Page 1 for whole docx
        else:
            logger_param.warning(f"Unsupported file type for basic parsing: {doc_path}")
            return None
        return content_list
    except ImportError as ie:
        logger_param.error(f"Import error during basic parsing of {doc_path}: {ie}. Ensure PyPDF2 and python-docx are installed.")
        return None
    except Exception as e:
        logger_param.error(f"Error in basic parse_document_text for {doc_path}: {e}")
        return None