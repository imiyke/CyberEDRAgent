import os
import sys

# Ensure repository root is on sys.path regardless of execution method
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import pytest
from tools.vector_store import VectorStoreManager
from config.llm_factory import load_config


TEST_QUERIES = [
    # 1. Raw obfuscated command (PowerShell / Command Execution)
    "powershell.exe -nop -w hidden -enc JABzAD0ATgBlAHcALQBPAG...",
    # 2. Raw SSH log (Brute Force / Initial Access)
    "sshd[1234]: Failed password for root from 192.168.1.50 port 55432 ssh2",
    # 3. Known tool execution (OS Credential Dumping)
    "mimikatz.exe privilege::debug sekurlsa::logonpasswords exit",
    # 4. Semantic query - simulating the LangGraph 'query_generator_node' output
    "Create scheduled task for persistence using schtasks.exe",
    # 5. Explicit MITRE ID query
    "Investigating attack technique T1059.001 PowerShell execution"
]


def is_db_available() -> bool:
    try:
        vsm = VectorStoreManager(auto_populate=False)
        return vsm.is_populated()
    except Exception:
        return False



def test_vector_store_manager_init():
    """Test that VectorStoreManager initializes with config and auto-populates if needed."""
    vsm = VectorStoreManager(auto_populate=True)
    assert vsm.vector_store is not None
    assert vsm.persist_directory is not None
    assert vsm.collection_name is not None
    assert vsm.is_populated()


@pytest.mark.parametrize("query", TEST_QUERIES)
def test_rag_retrieval(query: str):
    """Test standard similarity retrieval for various queries."""
    vsm = VectorStoreManager(auto_populate=True)
    results = vsm.similarity_search(query, k=3)
    assert isinstance(results, list)
    assert len(results) > 0
    for doc in results:
        assert hasattr(doc, "page_content")
        assert len(doc.page_content) > 0


def test_hybrid_search_mitre_id_extraction():
    """Test hybrid search with explicit MITRE technique ID."""
    vsm = VectorStoreManager(auto_populate=True)
    query = "Investigating T1059.001 command and script interpreter"
    results = vsm.hybrid_search(query, k=3)
    assert isinstance(results, list)
    assert len(results) > 0
    
    chunks = vsm.get_relevant_chunks(query, k=3)
    assert isinstance(chunks, list)
    assert len(chunks) == len(results)



def main():
    if not is_db_available():
        print("[!] Database not found. Run the ingestion script first (e.g. via app.py --reload-mitre-db).")
        return

    print("\n" + "=" * 50)
    print("🚀 RUNNING RAG RETRIEVAL TESTS (VectorStoreManager)")
    print("=" * 50)

    vsm = VectorStoreManager()

    for i, query in enumerate(TEST_QUERIES, 1):
        print(f"\n[Test {i}] Query: {query}")
        results = vsm.hybrid_search(query, k=3)
        
        for rank, doc in enumerate(results, 1):
            mitre_id = doc.metadata.get("mitre_id", "UNKNOWN") if doc.metadata else "UNKNOWN"
            name = doc.metadata.get("name", "Unknown Technique") if doc.metadata else "Unknown Technique"
            print(f"  {rank}. {mitre_id} - {name}")
            snippet = doc.page_content.replace("\n", " ")[:100]
            print(f"     Snippet: {snippet}...")


if __name__ == "__main__":
    main()