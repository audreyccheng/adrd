"""
JAR build wrapper for Calcite RuleSelector.

Provides:
- sync_source(): Write RuleSelector.java to canonical location
- rebuild_jar(): Run rebuild_jar.sh
- sync_and_rebuild(): Combined sync + rebuild
"""

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Tuple

logger = logging.getLogger(__name__)


def sync_source(ruleselector_java: str, canonical_path: str) -> None:
    """Write RuleSelector.java content to the canonical source location.

    Args:
        ruleselector_java: Java source code string for RuleSelector.java
        canonical_path: Path to the canonical RuleSelector.java file
    """
    path = Path(canonical_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(ruleselector_java)
    logger.info("Synced RuleSelector.java to %s (%d chars)", canonical_path, len(ruleselector_java))


def rebuild_jar(rebuild_script: str, timeout_sec: int = 120) -> Tuple[bool, str]:
    """Run the rebuild_jar.sh script to compile and update the JAR.

    Args:
        rebuild_script: Path to rebuild_jar.sh
        timeout_sec: Maximum time for the build

    Returns:
        (success, output_message) tuple
    """
    script = Path(rebuild_script)
    if not script.exists():
        return False, f"rebuild script not found: {rebuild_script}"

    try:
        result = subprocess.run(
            ["bash", str(script)],
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            cwd=script.parent,
        )

        if result.returncode == 0:
            logger.info("JAR rebuilt successfully")
            return True, result.stdout
        else:
            logger.error("JAR rebuild failed: %s", result.stderr[:500])
            return False, result.stderr[:500]

    except subprocess.TimeoutExpired:
        logger.error("JAR rebuild timed out after %ds", timeout_sec)
        return False, f"Build timed out after {timeout_sec}s"
    except Exception as e:
        logger.error("JAR rebuild error: %s", e)
        return False, str(e)


def sync_and_rebuild(
    ruleselector_java: str,
    canonical_path: str,
    rebuild_script: str,
    timeout_sec: int = 120,
) -> Tuple[bool, str]:
    """Sync RuleSelector.java source and rebuild the JAR.

    Args:
        ruleselector_java: Java source code string
        canonical_path: Path to canonical RuleSelector.java
        rebuild_script: Path to rebuild_jar.sh
        timeout_sec: Build timeout

    Returns:
        (success, message) tuple
    """
    sync_source(ruleselector_java, canonical_path)
    return rebuild_jar(rebuild_script, timeout_sec)
