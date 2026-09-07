import re
import logging
from typing import Any, Dict, Optional, Union
from langchain_core.messages import BaseMessage

logger = logging.getLogger("utils")

THINKING_KEYS = (
    "reasoning_content",
    "thinking",
    "thought",
    "reasoning",
    "reason",
    "reasoning_tokens"
)


def _extract_thinking_from_obj(obj: Any, depth: int = 0) -> Optional[str]:
    """
    Recursively scans dictionaries, lists, and objects for any thinking/reasoning field.
    Handles nested Ollama response structures like response_metadata['message']['thinking'].
    """
    if depth > 6 or obj is None:
        return None

    if isinstance(obj, dict):
        # 1. Direct key lookup
        for key in THINKING_KEYS:
            val = obj.get(key)
            if val is not None:
                if isinstance(val, str) and val.strip():
                    return val.strip()
                elif isinstance(val, list):
                    parts = []
                    for item in val:
                        if isinstance(item, str):
                            parts.append(item)
                        elif isinstance(item, dict):
                            sub = item.get("text") or item.get("thinking") or item.get("thought")
                            if sub:
                                parts.append(str(sub))
                    if parts:
                        return "\n".join(parts).strip()

        # 2. Check nested dicts / lists
        for k, v in obj.items():
            res = _extract_thinking_from_obj(v, depth + 1)
            if res:
                return res

    elif isinstance(obj, list):
        for item in obj:
            if isinstance(item, dict):
                # Check for Anthropic style content blocks: {"type": "thinking", "thinking": "..."}
                block_type = item.get("type", "")
                if block_type in ("thinking", "thought", "reasoning"):
                    think_val = item.get("thinking") or item.get("thought") or item.get("text")
                    if think_val:
                        return str(think_val).strip()
            res = _extract_thinking_from_obj(item, depth + 1)
            if res:
                return res

    return None


def parse_llm_response(response: Union[BaseMessage, str, Dict[str, Any], Any]) -> Dict[str, Optional[str]]:
    """
    Parse the LLM output and return a dictionary with:
         {"output": "text_output", "thinking": "llm_thinking" }

    Supports all major LLM and runtime formats:
    - Ollama (including response_metadata['message']['thinking'] and additional_kwargs)
    - DeepSeek-R1 / Qwen 2.5 & 3.5 reasoning tokens
    - Anthropic Claude 3.7 Extended Thinking (content blocks)
    - Google Gemini / Gemma 4 thought parts & metadata
    - Raw inline XML tags: <think>...</think>, <thought>...</thought>, <reasoning>...</reasoning>
    - Truncated / unclosed <think> blocks
    """
    thinking: Optional[str] = None
    output: str = ""

    # Case 1: BaseMessage / AIMessage from LangChain
    if isinstance(response, BaseMessage) or hasattr(response, "content"):
        # Step A: Deep recursive search in additional_kwargs & response_metadata
        additional_kwargs = getattr(response, "additional_kwargs", {}) or {}
        response_metadata = getattr(response, "response_metadata", {}) or {}
        generation_info = getattr(response, "generation_info", {}) or {}

        thinking = (
            _extract_thinking_from_obj(additional_kwargs)
            or _extract_thinking_from_obj(response_metadata)
            or _extract_thinking_from_obj(generation_info)
        )

        # Step B: Content inspection
        content = response.content
        if isinstance(content, list):
            text_parts = []
            for block in content:
                if isinstance(block, dict):
                    block_type = block.get("type", "")
                    if block_type in ("thinking", "thought", "reasoning"):
                        if not thinking:
                            thinking = block.get("thinking") or block.get("thought") or block.get("text", "")
                    elif block_type == "text" or "text" in block:
                        text_parts.append(block.get("text", ""))
                    elif "output" in block:
                        text_parts.append(block.get("output", ""))
                elif isinstance(block, str):
                    text_parts.append(block)
            output = "".join(text_parts).strip()
        elif isinstance(content, str):
            output = content
        else:
            output = str(content) if content is not None else ""

    # Case 2: Dictionary input (e.g. raw JSON/API payload)
    elif isinstance(response, dict):
        thinking = _extract_thinking_from_obj(response)
        output = str(response.get("output") or response.get("content") or response.get("text") or "")

    # Case 3: Plain string or fallback
    else:
        output = str(response) if response is not None else ""

    # Step C: Extract inline tags (<think>, <thought>, <reasoning>) if present in text
    thinking_patterns = [
        r"<think>(.*?)</think>",
        r"<thought>(.*?)</thought>",
        r"<reasoning>(.*?)</reasoning>",
        r"<reasoning_content>(.*?)</reasoning_content>"
    ]

    for pattern in thinking_patterns:
        match = re.search(pattern, output, flags=re.DOTALL | re.IGNORECASE)
        if match:
            extracted_think = match.group(1).strip()
            if not thinking:
                thinking = extracted_think
            elif extracted_think and len(extracted_think) > len(thinking):
                thinking = extracted_think
            
            # Remove the thinking tag block from the clean output
            output = re.sub(pattern, "", output, flags=re.DOTALL | re.IGNORECASE).strip()

    # Step D: Handle unclosed / truncated <think> tags (e.g. small model stuck in reasoning)
    for tag in ("<think>", "<thought>", "<reasoning>", "<reasoning_content>"):
        if tag in output.lower():
            parts = re.split(re.escape(tag), output, flags=re.IGNORECASE)
            # Text before the tag is clean output (if any)
            output = parts[0].strip()
            # Everything after the unclosed tag is reasoning
            if len(parts) > 1 and parts[1].strip():
                unclosed_think = parts[1].strip()
                if not thinking:
                    thinking = unclosed_think
                elif unclosed_think and len(unclosed_think) > len(thinking):
                    thinking = unclosed_think

    return {
        "output": output.strip(),
        "thinking": thinking.strip() if thinking and thinking.strip() else None
    }


