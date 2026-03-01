"""
Pattern implementer: uses Claude API to generate Java code for new patterns.

Phase 3 of the evolution loop.
"""

import json
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from evolve_loop.config import EvolutionConfig

logger = logging.getLogger(__name__)

# Accumulated API token usage for this module
_api_usage = {"input_tokens": 0, "output_tokens": 0}

_SYSTEM_PROMPT = None


def _load_system_prompt() -> str:
    """Load the implementer system prompt from the prompts directory."""
    global _SYSTEM_PROMPT
    if _SYSTEM_PROMPT is None:
        prompt_path = Path(__file__).parent / "prompts" / "implementer_system.md"
        _SYSTEM_PROMPT = prompt_path.read_text()
    return _SYSTEM_PROMPT


def _build_user_message(
    hypotheses: List[Dict],
    ruleselector_code: str,
    queryanalyzer_code: str,
) -> str:
    """Build the user message for the implementer Claude API call."""
    hyp_json = json.dumps(hypotheses, indent=2)

    return f"""## Pattern Hypotheses to Implement

```json
{hyp_json}
```

## Current RuleSelector.java

```java
{ruleselector_code}
```

## QueryAnalyzer.java (reference for available feature methods)

```java
{queryanalyzer_code}
```

## Instructions

1. Add new patterns for each hypothesis to RuleSelector.java
2. Place new patterns BEFORE the `// DEFAULT: No rules` comment
3. Place new patterns AFTER the last existing active pattern
4. Use tight feature conditions from each hypothesis
5. Include evidence comments
6. Every FSQ pattern MUST include `!selfJoinSubquery`
7. Return the COMPLETE modified RuleSelector.java (not a diff)
"""


def _try_compile(
    java_code: str,
    config: EvolutionConfig,
) -> Tuple[bool, str]:
    """Attempt to compile the Java code.

    Returns (success, error_message).
    """
    jar_dir = Path(config.jar_dir)
    jar_files = list(jar_dir.glob("*.jar"))
    if not jar_files:
        return False, f"No JAR files found in {jar_dir}"

    # Write code to a temp directory matching the expected package structure
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        # Create package directory structure
        pkg_dir = tmp / "org" / "apache" / "calcite" / "plan" / "hep"
        pkg_dir.mkdir(parents=True)

        # Write the RuleSelector.java
        rs_file = pkg_dir / "RuleSelector.java"
        rs_file.write_text(java_code)

        # Copy QueryAnalyzer.java if it exists in the JAR directory
        qa_src = jar_dir / "org" / "apache" / "calcite" / "plan" / "hep" / "QueryAnalyzer.java"
        if qa_src.exists():
            (pkg_dir / "QueryAnalyzer.java").write_text(qa_src.read_text())

        # Extract class files from JAR for compilation classpath
        classes_dir = tmp / "classes"
        classes_dir.mkdir()

        for jar in jar_files:
            subprocess.run(
                ["unzip", "-q", "-o", str(jar), "*.class"],
                cwd=str(classes_dir),
                capture_output=True,
            )

        # Compile
        result = subprocess.run(
            [
                "javac", "--release", "17", "-proc:none",
                "-cp", str(classes_dir),
                "-d", str(tmp / "out"),
                str(rs_file),
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )

        if result.returncode == 0:
            return True, ""
        else:
            return False, result.stderr[:1000]


def implement_patterns(
    hypotheses: List[Dict],
    ruleselector_code: str,
    queryanalyzer_code: str,
    config: EvolutionConfig,
    max_retries: int = 3,
) -> Optional[str]:
    """Generate Java code for new patterns using Claude API.

    Args:
        hypotheses: Pattern hypotheses from the analyst
        ruleselector_code: Current RuleSelector.java source code
        queryanalyzer_code: QueryAnalyzer.java source code
        config: Evolution config
        max_retries: Number of compilation retry attempts

    Returns:
        Modified RuleSelector.java source code, or None on failure
    """
    import anthropic

    if not config.anthropic_api_key:
        raise ValueError("ANTHROPIC_API_KEY not set in config or environment")

    if not hypotheses:
        logger.info("No hypotheses to implement")
        return None

    system_prompt = _load_system_prompt()
    user_message = _build_user_message(
        hypotheses, ruleselector_code, queryanalyzer_code
    )

    client = anthropic.Anthropic(api_key=config.anthropic_api_key)
    messages = [{"role": "user", "content": user_message}]

    for attempt in range(1, max_retries + 1):
        logger.info(
            "Calling Claude API for implementation (attempt %d/%d, model=%s)...",
            attempt, max_retries, config.model,
        )

        with client.messages.stream(
            model=config.model,
            max_tokens=config.max_tokens,
            system=system_prompt,
            messages=messages,
        ) as stream:
            response = stream.get_final_message()

        response_text = response.content[0].text
        _api_usage["input_tokens"] += response.usage.input_tokens
        _api_usage["output_tokens"] += response.usage.output_tokens
        logger.info(
            "Implementer response: %d chars, %d input/%d output tokens",
            len(response_text),
            response.usage.input_tokens,
            response.usage.output_tokens,
        )

        # Extract Java code from response
        java_code = _extract_java_code(response_text)
        if java_code is None:
            logger.warning("Could not extract Java code from response")
            messages.append({"role": "assistant", "content": response_text})
            messages.append({
                "role": "user",
                "content": "Your response did not contain a complete Java file. "
                "Please return the COMPLETE RuleSelector.java source code "
                "wrapped in ```java ... ``` markers.",
            })
            continue

        # Validate: must contain EVOLVE-BLOCK markers
        if "EVOLVE-BLOCK-START" not in java_code:
            logger.warning("Generated code missing EVOLVE-BLOCK-START marker")
            messages.append({"role": "assistant", "content": response_text})
            messages.append({
                "role": "user",
                "content": "The generated code is missing the EVOLVE-BLOCK-START marker. "
                "Please preserve ALL existing markers and code structure.",
            })
            continue

        # Try to compile
        success, error = _try_compile(java_code, config)
        if success:
            logger.info("Generated code compiles successfully")
            return java_code

        logger.warning("Compilation failed (attempt %d): %s", attempt, error[:200])
        messages.append({"role": "assistant", "content": response_text})
        messages.append({
            "role": "user",
            "content": f"Compilation failed with error:\n```\n{error}\n```\n"
            "Please fix the compilation error and return the complete corrected "
            "RuleSelector.java.",
        })

    logger.error("Failed to generate compilable code after %d attempts", max_retries)
    return None


def _extract_java_code(response_text: str) -> Optional[str]:
    """Extract Java source code from Claude's response."""
    from evolve_loop.utils.code_extraction import extract_java_code
    return extract_java_code(response_text)
