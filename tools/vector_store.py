import os
import shutil
import re
import logging
from typing import Optional, List, Tuple
from langchain_chroma import Chroma
from langchain_core.documents import Document
from config.llm_factory import get_embeddings, load_config

logger = logging.getLogger("vector_store")


class VectorStoreManager:
    """
    Manages access, auto-population, and retrieval for the Chroma MITRE ATT&CK vector store.
    """
    def __init__(
        self,
        collection_name: Optional[str] = None,
        persist_directory: Optional[str] = None,
        config: Optional[dict] = None,
        auto_populate: bool = True,
        force_reload: bool = False
    ):
        self.config = config if config is not None else load_config()
        vs_cfg = self.config.get("vector_store", {})
        
        self.persist_directory = persist_directory or vs_cfg.get("persist_directory", "./data/chroma_mitre_db")
        self.collection_name = collection_name or vs_cfg.get("collection_name", "mitre_knowledge_base")
        self.embeddings = get_embeddings(self.config)
        
        # Initialize Chroma instance and auto-populate if the collection is empty
        if force_reload:
            self.vector_store = self._populate_database(force_reload=True)
        else:
            self.vector_store = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory
            )
            if auto_populate and not self.is_populated():
                self.vector_store = self._populate_database(force_reload=False)

    def is_populated(self) -> bool:
        """Checks if the Chroma vector store collection actually contains documents."""
        try:
            if hasattr(self, "vector_store") and self.vector_store is not None:
                return self.vector_store._collection.count() > 0
            return False
        except Exception:
            return False

    def _populate_database(self, force_reload: bool = False) -> Chroma:
        """Downloads/loads STIX data and populates the Chroma vector database."""
        from tools.retrieve_mitre import load_mitre_data, extract_techniques

        if force_reload and os.path.exists(self.persist_directory):
            print(f"[*] Force reload requested. Deleting existing DB at '{self.persist_directory}'...")
            shutil.rmtree(self.persist_directory)

        stix_data = load_mitre_data()
        raw_docs = extract_techniques(stix_data)

        print(f"[*] Populating vector database at '{self.persist_directory}' (collection: '{self.collection_name}')...")
        return Chroma.from_documents(
            documents=raw_docs,
            embedding=self.embeddings,
            persist_directory=self.persist_directory,
            collection_name=self.collection_name
        )



    def similarity_search(self, query: str, k: int = 3) -> List[Document]:
        """Performs a standard dense vector similarity search."""
        return self.vector_store.similarity_search(query, k=k)

    def similarity_search_with_score(self, query: str, k: int = 3) -> List[Tuple[Document, float]]:
        """Performs a dense vector similarity search returning distance scores."""
        return self.vector_store.similarity_search_with_score(query, k=k)

    def hybrid_search(self, query: str, k: int = 3) -> List[Document]:
        """
        Retrieves context using hybrid search:
        1. Extracts explicit MITRE technique IDs (e.g. T1059.001) from the query for exact metadata retrieval.
        2. Executes dense semantic similarity search.
        3. Merges and deduplicates results while preserving relevance ranking.
        """
        results: List[Document] = []
        seen_ids = set()

        # Step 1: Detect explicit MITRE IDs in the query (e.g., T1059, T1059.001)
        extracted_mitre_ids = set(re.findall(r"T\d{4}(?:\.\d{3})?", query, re.IGNORECASE))
        if extracted_mitre_ids:
            logger.debug("Hybrid search extracted MITRE IDs: %s", extracted_mitre_ids)
        
        for mid in extracted_mitre_ids:
            try:
                # Query Chroma collection directly for exact MITRE ID metadata match
                lookup = self.vector_store.get(where={"mitre_id": mid.upper()})
                if lookup and lookup.get("documents"):
                    for idx, content in enumerate(lookup["documents"]):
                        metadata = lookup["metadatas"][idx] if lookup.get("metadatas") else {}
                        doc_id = metadata.get("mitre_id", mid.upper())
                        if doc_id not in seen_ids:
                            results.append(Document(page_content=content, metadata=metadata))
                            seen_ids.add(doc_id)
                            logger.debug("Exact match found for %s: %s", mid, metadata.get("name"))
            except Exception as e:
                logger.debug("Metadata lookup for %s failed: %s", mid, e)

        # Step 2: Dense semantic similarity search
        semantic_docs = self.vector_store.similarity_search(query, k=k)
        logger.debug("Dense semantic search retrieved %d documents", len(semantic_docs))
        for doc in semantic_docs:
            doc_id = doc.metadata.get("mitre_id") or hash(doc.page_content)
            if doc_id not in seen_ids:
                results.append(doc)
                seen_ids.add(doc_id)

        return results[:k]

    def get_random_documents(self, k: int = 3, exclude_ids: Optional[List[str]] = None) -> List[Document]:
        """
        Retrieves k pseudo-random distractor MITRE documents from the store.
        Used for distractor robustness and adversarial context resilience training.
        """
        import random
        exclude_set = set(e.upper() for e in (exclude_ids or []))
        try:
            if not hasattr(self, "_cached_all_docs") or self._cached_all_docs is None:
                collection_data = self.vector_store._collection.get(include=["documents", "metadatas"])
                self._cached_all_docs = []
                if collection_data and collection_data.get("documents"):
                    for idx, content in enumerate(collection_data["documents"]):
                        meta = collection_data["metadatas"][idx] if collection_data.get("metadatas") else {}
                        self._cached_all_docs.append(Document(page_content=content, metadata=meta))

            if not self._cached_all_docs:
                return []

            candidates = [d for d in self._cached_all_docs if d.metadata.get("mitre_id", "").upper() not in exclude_set]
            if not candidates:
                candidates = self._cached_all_docs

            sample_size = min(k, len(candidates))
            return random.sample(candidates, sample_size)
        except Exception as e:
            logger.warning("Failed to sample random distractor MITRE docs: %s", e)
            return []

    def get_relevant_chunks(self, query: str, k: int = 3) -> List[str]:
        """
        Convenience method to retrieve the text chunks for LLM context injection.
        """
        docs = self.hybrid_search(query, k=k)
        return [doc.page_content for doc in docs]

    def hybrid_search_formatted(self, query: str, k: int = 3) -> str:
        """
        Retrieves context and returns a human-readable formatted string.
        """
        results = self.hybrid_search(query, k=k)
        if not results:
            return "No relevant CTI data found in the knowledge base."
            
        formatted_results = []
        for i, doc in enumerate(results):
            res = f"--- Source {i+1} ---\n{doc.page_content}\n"
            if doc.metadata:
                res += f"Metadata: {doc.metadata}\n"
            formatted_results.append(res)
            
        return "\n".join(formatted_results)

