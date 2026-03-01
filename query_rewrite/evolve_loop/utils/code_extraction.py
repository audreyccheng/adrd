"""Shared Java code extraction from LLM responses.

Used by both implementer.py and fixer.py to parse RuleSelector.java
source code from Claude's responses.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def extract_java_code(response_text: str) -> Optional[str]:
    """Extract Java source code from an LLM response.

    Three tiers of extraction:
    1. Fenced ```java ... ``` block (most reliable)
    2. Any fenced ``` ... ``` block containing 'class RuleSelector'
    3. Bare code starting with 'package ' (tightened fallback)

    Args:
        response_text: Raw LLM response text.

    Returns:
        Extracted Java source code, or None if extraction fails.
    """
    text = response_text.strip()

    # Tier 1: ```java ... ``` block containing RuleSelector
    if "```java" in text:
        try:
            start = text.index("```java") + 7
            end = text.index("```", start)
            if end > start:
                candidate = text[start:end].strip()
                if "class RuleSelector" in candidate:
                    return candidate
                # First block isn't RuleSelector — fall through to Tier 2
        except ValueError:
            # No closing ``` — take everything after ```java as fallback
            start = text.index("```java") + 7
            candidate = text[start:].strip()
            if "class RuleSelector" in candidate:
                logger.warning("Missing closing fence, using everything after ```java")
                return candidate

    # Tier 2: Any fenced block containing RuleSelector
    if "```" in text:
        parts = text.split("```")
        for i in range(1, len(parts), 2):
            code = parts[i].strip()
            # Remove language tag if present
            if code.startswith("java\n"):
                code = code[5:]
            elif code.startswith("java "):
                code = code[5:]
            if "class RuleSelector" in code:
                return code.strip()

    # Tier 3: Bare code — tightened to prevent prose leakage
    # Must START with 'package ' (not just contain it anywhere)
    if "class RuleSelector" in text:
        stripped = text.lstrip()
        if stripped.startswith("package "):
            # Sanity: must have balanced-ish braces (real Java, not prose)
            if stripped.count("{") >= 2 and stripped.rstrip().endswith("}"):
                return stripped
            else:
                logger.warning(
                    "Tier 3 match rejected: brace count=%d, ends_with_brace=%s",
                    stripped.count("{"), stripped.rstrip().endswith("}")
                )

    return None
