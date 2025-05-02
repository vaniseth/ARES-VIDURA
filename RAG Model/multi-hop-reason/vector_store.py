import os
import re
import logging
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from sklearn.metrics.pairwise import cosine_similarity

# Assuming llm_interface.py contains LLMInterface class
from llm_interface import LLMInterface
# Assuming data_processing.py contains parse_and_chunk_documents
from data_processing import parse_and_chunk_documents
import config # Import config for defaults

class VectorStore(ABC):
    """Abstract base class for vector stores."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self._is_ready = False

    @abstractmethod
    def load_or_build(self,
                      documents_path_pattern: str,
                      chunk_settings: Dict,
                      embedding_interface: LLMInterface) -> bool:
        """Loads existing vector data or builds it from documents."""
        pass

    @abstractmethod
    def query(self, query_embedding: List[float], top_k: int) -> List[Dict[str, Any]]:
        """Queries the vector store for relevant chunks."""
        pass

    def is_ready(self) -> bool:
        """Checks if the vector store is initialized and populated."""
        return self._is_ready

    def _build_index_internal(self,
                              documents_path_pattern: str,
                              chunk_settings: Dict,
                              embedding_interface: LLMInterface) -> Optional[List[Dict[str, Any]]]:
        """Internal helper to parse, chunk, and embed documents."""
        self.logger.info("Starting internal index build process...")
        all_chunks_info = parse_and_chunk_documents(
            documents_path_pattern=documents_path_pattern,
            chunk_strategy=chunk_settings.get('strategy', config.DEFAULT_CHUNK_STRATEGY),
            chunk_size=chunk_settings.get('size', config.DEFAULT_CHUNK_SIZE),
            chunk_overlap=chunk_settings.get('overlap', config.DEFAULT_CHUNK_OVERLAP),
            min_chunk_length=config.MIN_CHUNK_LENGTH,
            logger=self.logger
        )
        if not all_chunks_info:
            self.logger.error("No chunks extracted. Index build failed.")
            return None

        chunk_texts = [info["chunk_text"] for info in all_chunks_info]
        self.logger.info(f"Generating embeddings for {len(chunk_texts)} chunks...")
        all_embeddings = embedding_interface.batch_get_embeddings(
            chunk_texts, task_type="RETRIEVAL_DOCUMENT"
        )

        processed_chunks_data = []
        failed_embeddings = 0
        for i, chunk_info in enumerate(all_chunks_info):
            if i < len(all_embeddings) and all_embeddings[i] is not None and len(all_embeddings[i]) > 0:
                processed_chunks_data.append({**chunk_info, "embeddings": all_embeddings[i]})
            else:
                failed_embeddings += 1
                self.logger.warning(f"Failed embedding for chunk {i} from {chunk_info.get('document_name', 'N/A')}. Skipping.")

        if not processed_chunks_data:
            self.logger.error("No chunks successfully embedded. Index build failed.")
            return None

        self.logger.info(f"Successfully embedded {len(processed_chunks_data)} chunks ({failed_embeddings} failures).")
        return processed_chunks_data


# --- CSV/InMemory Implementation ---
class PandasVectorStore(VectorStore):
    """Vector store using Pandas DataFrame (for CSV or In-Memory)."""

    def __init__(self, db_path: Optional[str], in_memory: bool, logger: logging.Logger):
        super().__init__(logger)
        self.db_path = db_path
        self.in_memory = in_memory
        self.vector_db_df: Optional[pd.DataFrame] = None
        if not in_memory and not db_path:
             raise ValueError("db_path must be provided for CSVVectorStore")

    def _save_dataframe_to_csv(self):
        """Saves the DataFrame to CSV."""
        if self.in_memory or not self.db_path or self.vector_db_df is None or self.vector_db_df.empty:
            return
        try:
            self.logger.info(f"Saving vector database DataFrame to {self.db_path}...")
            df_copy = self.vector_db_df.copy()
            # Convert numpy arrays to lists for CSV
            if 'embeddings' in df_copy.columns and isinstance(df_copy['embeddings'].iloc[0], np.ndarray):
                df_copy['embeddings'] = df_copy['embeddings'].apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else [])

            dir_name = os.path.dirname(self.db_path)
            if dir_name and not os.path.exists(dir_name):
                os.makedirs(dir_name)
            df_copy.to_csv(self.db_path, index=False)
            self.logger.info(f"DataFrame successfully saved to {self.db_path}")
        except Exception as e:
            self.logger.exception(f"Error saving DataFrame to CSV at {self.db_path}: {e}")

    def _load_dataframe_from_csv(self) -> bool:
        """Loads the DataFrame from CSV."""
        if not self.db_path or not os.path.exists(self.db_path):
            return False
        try:
            self.logger.info(f"Loading vector database from {self.db_path}...")
            self.vector_db_df = pd.read_csv(self.db_path)
            if 'embeddings' in self.vector_db_df.columns and not self.vector_db_df.empty:
                first_emb = self.vector_db_df['embeddings'].iloc[0]
                if isinstance(first_emb, str):
                    self.logger.info("Converting string embeddings in CSV to numpy arrays...")
                    self.vector_db_df['embeddings'] = self.vector_db_df['embeddings'].apply(
                        lambda x: np.array(list(map(float, re.findall(r"-?\d+\.?\d*", x)))) if isinstance(x, str) and x else np.array([])
                    )
                elif isinstance(first_emb, list):
                    self.vector_db_df['embeddings'] = self.vector_db_df['embeddings'].apply(np.array)

                # Validate embeddings
                valid_embeddings = self.vector_db_df['embeddings'].apply(lambda x: isinstance(x, np.ndarray) and x.size > 0)
                if valid_embeddings.any():
                    self.logger.info(f"Vector database loaded successfully. Shape: {self.vector_db_df.shape}")
                    self._is_ready = True
                    return True
                else:
                     self.logger.warning("CSV loaded but contains no valid embeddings after conversion.")
                     self.vector_db_df = None
                     return False
            elif self.vector_db_df.empty:
                 self.logger.warning(f"Loaded CSV file '{self.db_path}' is empty.")
                 self.vector_db_df = None
                 return False
            else:
                self.logger.error(f"CSV file '{self.db_path}' is missing the 'embeddings' column.")
                self.vector_db_df = None
                return False
        except Exception as e:
            self.logger.exception(f"An unexpected error occurred during CSV DB loading: {e}")
            self.vector_db_df = None
            return False

    def load_or_build(self,
                      documents_path_pattern: str,
                      chunk_settings: Dict,
                      embedding_interface: LLMInterface) -> bool:
        """Loads or builds the CSV/InMemory database."""
        loaded = False
        if not self.in_memory:
             loaded = self._load_dataframe_from_csv()

        if loaded:
             self.logger.info("Successfully loaded existing vector data.")
             self._is_ready = True
             return True
        elif self.in_memory and self.db_path and os.path.exists(self.db_path):
             # In-memory case: try loading from CSV first if it exists
             if self._load_dataframe_from_csv():
                  self.logger.info("Successfully loaded existing vector data into memory from CSV.")
                  self._is_ready = True
                  return True
             else:
                  self.logger.info("Failed to load from CSV for in-memory store, proceeding to build.")


        # Build if not loaded or if in-memory without existing CSV
        self.logger.info("Existing data not found or invalid. Building index...")
        processed_chunks = self._build_index_internal(documents_path_pattern, chunk_settings, embedding_interface)

        if processed_chunks:
            self.vector_db_df = pd.DataFrame(processed_chunks)
            # Ensure embeddings are numpy arrays for querying
            self.vector_db_df['embeddings'] = self.vector_db_df['embeddings'].apply(np.array)
            self.logger.info(f"Index built successfully. Shape: {self.vector_db_df.shape}")
            self._is_ready = True
            if not self.in_memory:
                self._save_dataframe_to_csv() # Save if CSV type
            elif self.db_path:
                 # Optionally save even for in-memory if path provided
                 self.logger.info("Saving newly built in-memory index to CSV for future use.")
                 self._save_dataframe_to_csv()
            return True
        else:
            self.logger.error("Failed to build index.")
            self._is_ready = False
            return False

    def query(self, query_embedding: List[float], top_k: int) -> List[Dict[str, Any]]:
        """Retrieves relevant chunks from the DataFrame."""
        if not self.is_ready() or self.vector_db_df is None or self.vector_db_df.empty:
            self.logger.warning("Pandas vector store is not ready or empty.")
            return []

        query_embedding_2d = np.array([query_embedding])
        is_valid_embedding = self.vector_db_df['embeddings'].apply(lambda x: isinstance(x, np.ndarray) and x.size > 0)
        valid_df = self.vector_db_df[is_valid_embedding]

        if valid_df.empty:
             self.logger.error("No valid numpy array embeddings found in DataFrame.")
             return []

        try:
            database_embeddings = np.stack(valid_df["embeddings"].to_numpy())
            if database_embeddings.shape[1] != query_embedding_2d.shape[1]:
                self.logger.error(f"Embedding dimension mismatch: Query ({query_embedding_2d.shape[1]}) vs DB ({database_embeddings.shape[1]})")
                return []
        except Exception as stack_err:
            self.logger.error(f"Error stacking database embeddings: {stack_err}.")
            return []

        try:
            similarities = cosine_similarity(query_embedding_2d, database_embeddings)[0]
        except Exception as sim_err:
            self.logger.error(f"Error calculating cosine similarity: {sim_err}")
            return []

        actual_top_k = min(top_k, len(similarities))
        if actual_top_k <= 0: return []

        top_indices_local = np.argsort(similarities)[-actual_top_k:][::-1]
        top_original_indices = valid_df.index[top_indices_local]

        results = []
        relevant_rows = self.vector_db_df.loc[top_original_indices]
        for i, (original_idx, row) in enumerate(relevant_rows.iterrows()):
            similarity_score = similarities[top_indices_local[i]]
            results.append({
                 "id": f"chunk_{original_idx}",
                 "text": row['chunk_text'],
                 "metadata": {
                     "document_name": row.get('document_name', 'Unknown'),
                     "page_number": int(row.get('page_number', -1)),
                     "chunk_number": int(row.get('chunk_number', -1)),
                 },
                 "score": float(similarity_score)
             })
        return results

# --- ChromaDB Implementation ---
class ChromaVectorStore(VectorStore):
    """Vector store using ChromaDB."""

    def __init__(self, collection_name: str = "cnt_collection", persist_directory: Optional[str] = None, logger: logging.Logger = logging.getLogger("CNTRAG")):
        super().__init__(logger)
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.chroma_client = None
        self.collection = None
        self._initialize_chroma_client()

    def _initialize_chroma_client(self):
        try:
            import chromadb
            from chromadb.config import Settings

            client_settings = Settings()
            if self.persist_directory:
                os.makedirs(self.persist_directory, exist_ok=True)
                client_settings = Settings(persist_directory=self.persist_directory, is_persistent=True) # Correct way for persistence
                self.logger.info(f"Initializing ChromaDB with persistence at: {self.persist_directory}")
            else:
                 self.logger.info("Initializing ChromaDB in-memory.")


            self.chroma_client = chromadb.Client(client_settings)
            self.collection = self.chroma_client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"} # Explicitly use cosine distance
            )
            self.logger.info(f"ChromaDB client connected. Using collection: '{self.collection_name}'")
            # Check initial count to help determine if build is needed
            try:
                if self.collection.count() > 0:
                    self.logger.info(f"Chroma collection '{self.collection_name}' already contains {self.collection.count()} documents.")
                    self._is_ready = True # Assume ready if count > 0
            except Exception as count_err:
                 self.logger.warning(f"Could not get initial count from Chroma: {count_err}")


        except ImportError:
            self.logger.error("ChromaDB library not installed. Please run 'pip install chromadb'")
            raise # Re-raise to indicate failure
        except Exception as e:
            self.logger.exception(f"An unexpected error occurred during ChromaDB setup: {e}")
            raise # Re-raise

    def _populate_chroma(self, processed_chunks: List[Dict[str, Any]]):
        """Adds processed chunks (with embeddings) to the Chroma collection."""
        if not self.collection or not processed_chunks:
            self.logger.warning("Chroma collection not initialized or no chunks to populate.")
            return False

        ids = []
        embeddings = []
        documents = []
        metadatas = []

        for i, chunk_data in enumerate(processed_chunks):
            if chunk_data.get('embeddings') and chunk_data.get('chunk_text'):
                 ids.append(f"chunk_{i}_{chunk_data.get('document_name', 'doc')[:10]}") # More unique ID
                 embeddings.append(chunk_data['embeddings'])
                 documents.append(chunk_data['chunk_text'])
                 metadatas.append({
                     "document_name": chunk_data.get('document_name', 'Unknown'),
                     "page_number": int(chunk_data.get('page_number', -1)),
                     "chunk_number": int(chunk_data.get('chunk_number', -1)),
                 })

        if not ids:
            self.logger.warning("No valid chunks found to add to ChromaDB.")
            return False

        batch_size = 100
        total_added = 0
        try:
            for i in range(0, len(ids), batch_size):
                batch_ids = ids[i:i+batch_size]
                batch_embeddings = embeddings[i:i+batch_size]
                batch_documents = documents[i:i+batch_size]
                batch_metadatas = metadatas[i:i+batch_size]

                if not batch_ids: continue

                self.collection.add(
                    ids=batch_ids,
                    embeddings=batch_embeddings,
                    documents=batch_documents,
                    metadatas=batch_metadatas
                )
                total_added += len(batch_ids)
                self.logger.info(f"Added batch {i//batch_size + 1} ({len(batch_ids)} docs) to ChromaDB. Total added: {total_added}")

            # Optional: Persist changes if using disk-based Chroma
            if self.persist_directory:
                 self.logger.info("Persisting ChromaDB changes...")
                 # Chroma client with `is_persistent=True` handles persistence automatically on add/delete etc.
                 # Explicit persist call isn't usually needed with the Settings approach.
                 # If using older chromadb versions or manual persistence:
                 # self.chroma_client.persist()
                 pass


            self.logger.info(f"Finished populating ChromaDB. Added {total_added} documents.")
            return True
        except Exception as batch_err:
            self.logger.exception(f"Error adding batch to ChromaDB: {batch_err}. Population may be incomplete.")
            return False


    def load_or_build(self,
                      documents_path_pattern: str,
                      chunk_settings: Dict,
                      embedding_interface: LLMInterface) -> bool:
        """Builds the ChromaDB index if it's empty."""
        if not self.collection:
             self.logger.error("ChromaDB collection not initialized. Cannot load or build.")
             return False

        try:
            # Check count again, might have been populated after init
            current_count = self.collection.count()
            if current_count > 0:
                self.logger.info(f"ChromaDB collection '{self.collection_name}' already has {current_count} documents. Assuming loaded.")
                self._is_ready = True
                return True
        except Exception as count_err:
            self.logger.warning(f"Could not verify ChromaDB count: {count_err}. Attempting build if necessary.")


        self.logger.info(f"ChromaDB collection '{self.collection_name}' appears empty. Building index...")
        processed_chunks = self._build_index_internal(documents_path_pattern, chunk_settings, embedding_interface)

        if processed_chunks:
             populated = self._populate_chroma(processed_chunks)
             self._is_ready = populated
             return populated
        else:
            self.logger.error("Failed to build index (no processed chunks).")
            self._is_ready = False
            return False

    def query(self, query_embedding: List[float], top_k: int) -> List[Dict[str, Any]]:
        """Queries the ChromaDB collection."""
        if not self.is_ready() or not self.collection:
            self.logger.warning("ChromaDB collection is not ready or not initialized.")
            return []
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"] # Request distances
            )

            if not results or not results.get("ids") or not results["ids"][0]:
                self.logger.debug("No relevant chunks found in ChromaDB.")
                return []

            formatted_results = []
            ids = results["ids"][0]
            distances = results.get("distances", [[]])[0]
            documents = results.get("documents", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]

            for i, chunk_id in enumerate(ids):
                if i < len(distances) and i < len(documents) and i < len(metadatas):
                     distance = distances[i]
                     # Similarity for cosine distance: 1 - distance
                     # Similarity for L2 distance: depends, maybe exp(-distance) or 1 / (1 + distance)
                     # Assuming cosine space as configured
                     similarity_score = 1.0 - distance

                     formatted_results.append({
                         "id": chunk_id,
                         "text": documents[i],
                         "metadata": metadatas[i] if metadatas[i] else {},
                         "score": float(similarity_score)
                     })
                else:
                    self.logger.warning(f"Inconsistent result lengths from ChromaDB query. Skipping partial result at index {i}.")

            return formatted_results

        except Exception as e:
            self.logger.exception(f"Error querying ChromaDB: {e}")
            return []

# --- Factory Function ---
def get_vector_store(vector_db_type: str, vector_db_path: Optional[str], logger: logging.Logger) -> VectorStore:
    """Factory function to create the appropriate vector store instance."""
    db_type = vector_db_type.lower()
    if db_type == "csv":
        logger.info("Creating CSV Vector Store")
        if not vector_db_path:
            raise ValueError("vector_db_path is required for CSV vector store.")
        return PandasVectorStore(db_path=vector_db_path, in_memory=False, logger=logger)
    elif db_type == "inmemory":
        logger.info("Creating InMemory Vector Store")
        # Allow path for potential loading/saving cache even if in-memory
        return PandasVectorStore(db_path=vector_db_path, in_memory=True, logger=logger)
    elif db_type == "chroma":
        logger.info("Creating ChromaDB Vector Store")
        # Use path as persist directory if provided, otherwise in-memory
        persist_dir = vector_db_path if vector_db_path else None
        # Can customize collection name if needed
        return ChromaVectorStore(persist_directory=persist_dir, logger=logger)
    else:
        raise ValueError(f"Unknown vector database type: {vector_db_type}")
