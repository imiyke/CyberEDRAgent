import os
import yaml
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

def load_config(config_path: str = "config/config.yaml") -> dict:
    """Loads the YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def get_llm(role: str, config: dict):
    """
    Dynamically instantiate an LLM based on the role and configuration.
    
    If 'all' is defined in config['llms'], it overrides specific roles for testing.
    
    Args:
        role (str): The agent role (e.g., 'query_generator', 'synthesizer').
        config (dict): The loaded configuration dictionary.
    
    Returns:
        BaseChatModel: A Langchain ChatModel instance.
    """
    if "llms" not in config:
        raise ValueError("Missing 'llms' section in configuration.")
    
    llms_config = config["llms"]
    
    # Check if a global override 'all' is specified
    if "all" in llms_config and llms_config["all"]:
        role_config = llms_config["all"]
    elif role in llms_config:
        role_config = llms_config[role]
    elif role == "synthesizer" and "synthetizer" in llms_config:
        role_config = llms_config["synthetizer"]
    elif role == "synthetizer" and "synthesizer" in llms_config:
        role_config = llms_config["synthesizer"]
    else:
        raise ValueError(f"Role '{role}' not found in configuration.")
    
    provider = role_config.get("provider")
    model_name = role_config.get("model")
    
    if provider == "ollama":
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        num_ctx = role_config.get("num_ctx", 3072)
        num_predict = role_config.get("num_predict", 1024)
        return ChatOllama(
            model=model_name,
            base_url=base_url,
            temperature=0,
            num_ctx=num_ctx,
            num_predict=num_predict,
            reasoning=True
        )



    elif provider == "google":
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is missing.")
        return ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key, temperature=0)
    else:
        raise ValueError(f"Unsupported provider: {provider}")


def get_embeddings(config: dict):
    """
    Instantiate the configured embedding model.
    
    Args:
        config (dict): The loaded configuration dictionary.
    
    Returns:
        Embeddings: A Langchain Embeddings instance.
    """
    provider = config.get("embeddings", {}).get("provider", "ollama")
    model_name = config.get("embeddings", {}).get("model", "nomic-embed-text")
    
    if provider == "ollama":
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        return OllamaEmbeddings(model=model_name, base_url=base_url)
    else:
        raise NotImplementedError(f"Embeddings provider {provider} is not supported yet.")
