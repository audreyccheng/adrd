"""
Feature extraction wrapper around Calcite's QueryAnalyzer.

Provides:
- extract_features_for_query(): Extract features for a single query
- build_feature_matrix(): Build feature dict for all queries
"""

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from evolve_loop.utils.java_bridge import extract_features as _jb_extract_features

logger = logging.getLogger(__name__)


def load_query(query_dir: str, query_id: str) -> Tuple[str, str]:
    """Load a SQL query file and its CREATE TABLE statements.

    Args:
        query_dir: Base directory for the benchmark (e.g., benchmarks/tpch)
        query_id: Query identifier (e.g., "query17_0" or "query17_0_1")

    Returns:
        (sql, create_tables_str) tuple
    """
    base_dir = Path(query_dir)
    parts = query_id.split("_")
    query_base = parts[0]

    if len(parts) >= 2:
        file_name = f"{parts[0]}_{parts[1]}"
        stmt_idx = int(parts[2]) if len(parts) >= 3 else 0
    else:
        file_name = query_id
        stmt_idx = 0

    sql_file = base_dir / query_base / f"{file_name}.sql"
    if not sql_file.exists():
        raise FileNotFoundError(f"Query file not found: {sql_file}")

    content = sql_file.read_text()
    content = re.sub(r"--.*\n", "\n", content)
    statements = [s.strip() for s in content.split(";") if s.strip()]
    statements = [
        s for s in statements
        if s.upper().startswith("SELECT") or s.upper().startswith("WITH")
    ]

    sql = (
        statements[stmt_idx]
        if stmt_idx < len(statements)
        else statements[0] if statements else content.strip()
    )

    create_file = base_dir / "create_tables.sql"
    create_tables = create_file.read_text() if create_file.exists() else ""

    return sql, create_tables


def extract_features_for_query(
    sql: str,
    create_tables_str: str,
) -> Optional[Dict]:
    """Extract structural features from a SQL query.

    Args:
        sql: The SQL query
        create_tables_str: CREATE TABLE statements

    Returns:
        Dict of features, or None on failure
    """
    return _jb_extract_features(sql, create_tables_str)


def build_feature_matrix(
    queries: List[Tuple[str, str, str]],
) -> Dict[str, Dict]:
    """Build a feature matrix for a list of queries.

    Args:
        queries: List of (query_id, sql, create_tables_str) tuples

    Returns:
        Dict mapping query_id to feature dict
    """
    matrix = {}
    for query_id, sql, create_tables_str in queries:
        features = extract_features_for_query(sql, create_tables_str)
        if features is not None:
            matrix[query_id] = features
        else:
            logger.warning("Feature extraction failed for %s", query_id)
    return matrix


def discover_queries(query_dirs: Dict[str, str]) -> List[Tuple[str, str]]:
    """Discover all available query files.

    Args:
        query_dirs: Dict mapping benchmark name to query directory path

    Returns:
        List of (benchmark, query_id) tuples
    """
    queries = []
    for benchmark, dir_path in query_dirs.items():
        base = Path(dir_path)
        if not base.exists():
            logger.warning("Query directory not found: %s", dir_path)
            continue

        for query_dir in sorted(base.iterdir()):
            if not query_dir.is_dir() or query_dir.name.startswith("."):
                continue
            if query_dir.name == "create_tables.sql":
                continue

            for sql_file in sorted(query_dir.glob("*.sql")):
                # Extract query_id from filename (e.g., query17_0.sql → query17_0)
                query_id = sql_file.stem
                queries.append((benchmark, query_id))

    return queries
