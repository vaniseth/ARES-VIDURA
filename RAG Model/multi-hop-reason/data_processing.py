# data_processing.py
import os
import glob
import logging
import hashlib
from typing import List, Optional, Dict, Any, Tuple
from langchain_text_splitters import RecursiveCharacterTextSplitter
import PyPDF2
import config


from llm_interface import LLMInterface

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


def parse_and_chunk_documents(
    documents_path_pattern: str,
    chunk_strategy: str, # This will be 'recursive'
    chunk_size: int = config.DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = config.DEFAULT_CHUNK_OVERLAP,
    embedding_interface: Optional['LLMInterface'] = None, # Not used here, but kept for signature consistency
    logger_parent: logging.Logger = logging.getLogger("CNTRAG")
) -> List[Dict[str, Any]]:
    """
    Parses PDF documents using PyPDF2 and chunks them recursively.
    This version DOES NOT require the 'unstructured' library and creates detailed source metadata.
    """
    global logger
    logger = logger_parent

    logger.info(f"Starting document parsing with PyPDF2 (Strategy: 'recursive')")
    document_files = glob.glob(documents_path_pattern)
    document_files = [f for f in document_files if os.path.isfile(f) and not os.path.basename(f).startswith('~')]

    if not document_files:
        logger.error(f"No document files found matching pattern: {documents_path_pattern}")
        return []

    all_final_chunks_list: List[Dict[str, Any]] = []

    # Initialize the recursive splitter once
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""
        ] # Generic separators for recursive splitting
    )

    for doc_path in document_files:
        doc_name = os.path.basename(doc_path)
        
        if not doc_path.lower().endswith('.pdf'):
            logger.warning(f"Skipping non-PDF file: {doc_name}")
            continue
            
        logger.info(f"Processing document: {doc_name}")
        try:
            with open(doc_path, "rb") as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num, page in enumerate(pdf_reader.pages):
                    page_actual_number = page_num + 1
                    try:
                        page_text = page.extract_text()
                        if not page_text or len(page_text.strip()) < config.MIN_CHUNK_LENGTH:
                            continue
                    except Exception as page_err:
                        logger.error(f"Error extracting text from {doc_name}, page {page_actual_number}: {page_err}")
                        continue

                    # Split the text from the entire page
                    sub_chunks = splitter.split_text(page_text)

                    for sub_idx, chunk_text in enumerate(sub_chunks):
                        cleaned_chunk = chunk_text.strip()
                        if len(cleaned_chunk) < config.MIN_CHUNK_LENGTH:
                            continue

                        # Create a unique, deterministic ID for this specific chunk
                        hasher = hashlib.sha1()
                        hasher.update(cleaned_chunk.encode('utf-8'))
                        content_hash = hasher.hexdigest()[:10]
                        chunk_id = f"{os.path.splitext(doc_name)[0]}_p{page_actual_number}_c{sub_idx}_{content_hash}"

                        # Create the detailed metadata dictionary
                        chunk_metadata = {
                            "document_name": doc_name,
                            "page_number": page_actual_number,
                            "chunk_on_page": sub_idx + 1, # The index of this chunk on this specific page
                            "chunk_id": chunk_id,
                        }

                        all_final_chunks_list.append({
                            "chunk_text": cleaned_chunk,
                            "metadata": chunk_metadata
                        })

        except Exception as e:
            logger.exception(f"Failed to process PDF file {doc_path}: {e}")

    logger.info(f"Finished parsing. Total final chunks created: {len(all_final_chunks_list)}")
    return all_final_chunks_list


# Fallback basic parser (ensure it's defined or imported correctly)
def parse_document_text(doc_path: str, logger_param: logging.Logger) -> Optional[List[Tuple[int, str]]]:
    """Basic text extraction for PDF/DOCX (simplified fallback)."""
    logger_param.debug(f"Basic parse_document_text called for {doc_path}")
    content_list = []
    try:
        if doc_path.lower().endswith(".pdf"):
            import PyPDF2
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