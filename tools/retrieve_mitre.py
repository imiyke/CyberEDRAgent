import os
import shutil
import json
import requests
from typing import Optional
from langchain_core.documents import Document
from langchain_chroma import Chroma
from config.llm_factory import get_embeddings, load_config

# Configuration defaults
MITRE_STIX_URL = "https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json"
LOCAL_BACKUP_PATH = "./data/enterprise-attack.json.bckup"


def load_mitre_data(backup_path: str = LOCAL_BACKUP_PATH) -> dict:
    """
    Fetches the latest Enterprise ATT&CK matrix in STIX format.
    Falls back to a local backup if offline or download fails.
    """
    try:
        print("[*] Downloading MITRE ATT&CK STIX data from GitHub...")
        response = requests.get(MITRE_STIX_URL, timeout=30)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"[!] Failed to download STIX data ({e}). Checking local backup at '{backup_path}'...")
        if os.path.exists(backup_path):
            with open(backup_path, "r", encoding="utf-8") as f:
                print(f"[+] Loaded STIX data from local backup '{backup_path}'.")
                return json.load(f)
        raise RuntimeError(f"Could not load MITRE STIX data from URL or local backup '{backup_path}'.") from e


def extract_techniques(stix_data: dict) -> list[Document]:
    """Parses STIX objects to extract active MITRE techniques."""
    print("[*] Extracting attack patterns...")
    documents = []
    
    for obj in stix_data.get("objects", []):
        if obj.get("type") != "attack-pattern":
            continue
        if obj.get("revoked") or obj.get("x_mitre_deprecated"):
            continue
            
        mitre_id = None
        for ref in obj.get("external_references", []):
            if ref.get("source_name") == "mitre-attack":
                mitre_id = ref.get("external_id")
                break
                
        if not mitre_id:
            continue
            
        name = obj.get("name", "Unknown")
        description = obj.get("description", "")
        
        # Structure the content. With Nomic's 8K context, we keep the full description intact.
        content = f"MITRE ID: {mitre_id}\nTechnique Name: {name}\nDescription: {description}"
        
        doc = Document(
            page_content=content,
            metadata={"mitre_id": mitre_id, "name": name}
        )
        documents.append(doc)
        
    print(f"[+] Extracted {len(documents)} techniques.")
    return documents


def retrieve_mitre_db(force_reload: bool = False, config: Optional[dict] = None) -> Chroma:
    """
    Ensures the Chroma vector database is populated with MITRE ATT&CK techniques.
    Delegates to VectorStoreManager.
    """
    from tools.vector_store import VectorStoreManager
    vsm = VectorStoreManager(config=config, auto_populate=True, force_reload=force_reload)
    return vsm.vector_store


