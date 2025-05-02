import os
import glob
import logging
import PyPDF2
import docx # Requires python-docx
from typing import List, Tuple, Optional, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config # Import config for defaults

def parse_document_text(doc_path: str, logger: logging.Logger) -> Optional[List[Tuple[int, str]]]:
    """
    Extracts text.
    For PDF: Returns list of (page_num, page_text).
    For DOCX: Returns list containing [(1, full_doc_text)].
    """
    doc_name = os.path.basename(doc_path)
    content_list = []
    try:
        if doc_path.lower().endswith(".pdf"):
            with open(doc_path, "rb") as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text and len(page_text.strip()) > 10:
                            content_list.append((page_num + 1, page_text))
                        else:
                            logger.debug(f"Skipping empty/short page {page_num + 1} in PDF {doc_name}")
                    except Exception as page_err:
                        logger.error(f"Error extracting text from PDF {doc_name}, page {page_num + 1}: {page_err}")
        elif doc_path.lower().endswith(".docx"):
            document = docx.Document(doc_path)
            full_text = "\n".join([p.text for p in document.paragraphs if p.text.strip()])
            if full_text and len(full_text.strip()) > 10:
                content_list.append((1, full_text)) # Use 1 as the "item number" for the whole doc
            else:
                 logger.debug(f"Skipping empty/short DOCX {doc_name}")
        else:
            logger.warning(f"Unsupported document format: {doc_path}. Skipping.")
            return None
        return content_list
    except FileNotFoundError:
        logger.error(f"Document not found during parsing: {doc_path}")
        return None
    except Exception as e:
        logger.exception(f"Error processing document {doc_path}: {e}")
        return None


def parse_and_chunk_documents(
    documents_path_pattern: str,
    chunk_strategy: str = config.DEFAULT_CHUNK_STRATEGY,
    chunk_size: int = config.DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = config.DEFAULT_CHUNK_OVERLAP,
    min_chunk_length: int = config.MIN_CHUNK_LENGTH,
    logger: logging.Logger = logging.getLogger("CNTRAG")
) -> List[Dict[str, Any]]:
    """Parses PDFs/DOCX and splits them into chunks."""
    logger.info(f"Parsing documents using pattern: {documents_path_pattern}")
    document_files = glob.glob(documents_path_pattern)
    document_files = [f for f in document_files if os.path.isfile(f)]

    if not document_files:
        logger.error(f"No document files found matching pattern: {documents_path_pattern}")
        return []
    logger.info(f"Found {len(document_files)} potential document files to process.")

    if chunk_strategy == "recursive":
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""],
            length_function=len,
            is_separator_regex=False,
        )
    else:
        logger.warning(f"Unknown chunk strategy '{chunk_strategy}'. Defaulting to 'recursive'.")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]
        )

    all_chunks_info = []
    total_chunks_processed = 0

    for doc_path in document_files:
        logger.debug(f"Processing document: {doc_path}")
        doc_name = os.path.basename(doc_path)
        doc_content = parse_document_text(doc_path, logger)

        if not doc_content:
            logger.debug(f"No content extracted from {doc_path}. Skipping.")
            continue

        for item_num, item_text in doc_content:
            try:
                chunks = text_splitter.split_text(item_text)
                for chunk_num, chunk_text in enumerate(chunks):
                    cleaned_chunk = chunk_text.strip()
                    if len(cleaned_chunk) >= min_chunk_length:
                        all_chunks_info.append({
                            "document_name": doc_name,
                            "page_number": item_num, # Page (PDF) or 1 (DOCX)
                            "chunk_number": chunk_num,
                            "chunk_text": cleaned_chunk,
                        })
                        total_chunks_processed += 1
                    else:
                        logger.debug(f"Skipping short/noisy chunk {chunk_num} from item {item_num} in {doc_name}")
            except Exception as split_err:
                logger.error(f"Error splitting text for item {item_num} of {doc_name}: {split_err}")

    logger.info(f"Finished parsing. Extracted {len(all_chunks_info)} valid chunks from {len(document_files)} documents.")
    return all_chunks_info
