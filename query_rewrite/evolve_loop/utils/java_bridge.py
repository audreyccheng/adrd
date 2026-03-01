"""
JPype bridge to Calcite's Java-based query rewriter and feature extractor.

Provides:
- init_jvm(): Start JVM with Calcite JARs
- rewrite_sql(): Apply a rule combination to SQL
- extract_features(): Extract query structural features via QueryAnalyzer
"""

import os
import re
import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_jvm_initialized = False
_Rewriter = None
_QueryAnalyzer = None


def init_jvm(jar_dir: str) -> None:
    """Start the JVM and load Calcite classes.

    Args:
        jar_dir: Path to directory containing the LearnedRewrite JAR and dependencies.
    """
    global _jvm_initialized, _Rewriter, _QueryAnalyzer
    if _jvm_initialized:
        return

    import jpype as jp

    jar_path = Path(jar_dir)
    jars = list(jar_path.glob("*.jar"))
    if not jars:
        raise FileNotFoundError(f"No JAR files found in {jar_dir}")
    classpath = [str(j) for j in jars]

    if not jp.isJVMStarted():
        # Use JAVA_HOME if set, otherwise try to find Java 21+
        jvm_path = None
        java_home = os.environ.get("JAVA_HOME")
        if java_home:
            candidate = Path(java_home) / "lib" / "server" / "libjvm.dylib"
            if candidate.exists():
                jvm_path = str(candidate)
            else:
                candidate = Path(java_home) / "lib" / "libjli.dylib"
                if candidate.exists():
                    jvm_path = str(candidate)

        if jvm_path is None:
            jvm_path = jp.getDefaultJVMPath()

        jp.startJVM(jvm_path, classpath=classpath)

    import jpype.imports  # noqa: F401 - enables Java imports
    from rewriter import Rewriter
    _Rewriter = Rewriter
    _jvm_initialized = True
    logger.info("JVM initialized with %d JARs from %s", len(jars), jar_dir)


def _to_java_list(items: List[str]):
    """Convert a Python list of strings to a Java ArrayList."""
    import jpype as jp
    ArrayList = jp.JClass("java.util.ArrayList")
    jlist = ArrayList()
    for item in items:
        jlist.add(jp.JString(item))
    return jlist


def _parse_create_tables(create_tables_str: str) -> List[str]:
    """Parse a create_tables string into individual CREATE TABLE statements."""
    stmts = []
    if not create_tables_str:
        return stmts
    for stmt in create_tables_str.split(";"):
        stmt = stmt.strip()
        if stmt.upper().startswith("CREATE TABLE"):
            stmts.append(stmt + ";")
    return stmts


def rewrite_sql(
    sql: str,
    create_tables_str: str,
    rules: List[str],
) -> Optional[str]:
    """Apply a set of Calcite rewrite rules to SQL.

    Args:
        sql: The original SQL query.
        create_tables_str: CREATE TABLE statements (semicolon-separated).
        rules: List of Calcite rule names to apply.

    Returns:
        Rewritten SQL string, or None if rewriting fails or produces no change.
    """
    if not _jvm_initialized:
        raise RuntimeError("JVM not initialized. Call init_jvm() first.")

    # Empty rules means no rewrite requested.
    # Note: Rewriter treats empty list as "use RuleSelector", so we short-circuit.
    if not rules:
        return None

    import jpype as jp

    try:
        create_tables = _parse_create_tables(create_tables_str)
        result = _Rewriter.rewrite(
            jp.JString(sql),
            _to_java_list(create_tables),
            _to_java_list(rules),
            jp.JInt(len(rules)),
            jp.JString("postgresql"),
        )

        if result and hasattr(result, "sql"):
            rewritten = str(result.sql)
            # Return None if SQL didn't actually change
            # Normalize whitespace for comparison since Calcite may reformat
            orig_norm = " ".join(sql.split()).lower()
            new_norm = " ".join(rewritten.split()).lower()
            if orig_norm == new_norm:
                return None
            return rewritten
        return None
    except Exception as e:
        logger.debug("Rewrite failed for rules %s: %s", rules, e)
        return None


def extract_features(sql: str, create_tables_str: str) -> Optional[Dict]:
    """Extract structural features from a SQL query using QueryAnalyzer.

    This calls the Java-side QueryAnalyzer which parses SQL into a Calcite
    RelNode tree and extracts features like join count, subquery presence, etc.

    Args:
        sql: The SQL query to analyze.
        create_tables_str: CREATE TABLE statements for schema context.

    Returns:
        Dict of features, or None if extraction fails.
    """
    if not _jvm_initialized:
        raise RuntimeError("JVM not initialized. Call init_jvm() first.")

    import jpype as jp

    try:
        create_tables = _parse_create_tables(create_tables_str)

        # Use Rewriter to parse SQL into RelNode, then call QueryAnalyzer
        # Rewriter.rewrite with empty rules parses SQL and returns r1 = original RelNode
        result = _Rewriter.rewrite(
            jp.JString(sql),
            _to_java_list(create_tables),
            _to_java_list([]),  # No rules - just parse
            jp.JInt(0),
            jp.JString("postgresql"),
        )

        if not result or not hasattr(result, "r1") or result.r1 is None:
            logger.debug("Could not parse SQL for feature extraction")
            return None

        root = result.r1
        QA = jp.JClass("org.apache.calcite.plan.hep.QueryAnalyzer")
        mq = None  # MetadataQuery not needed for feature extraction

        features = {
            # --- Original 15 features ---
            "joins": int(QA.countJoins(root)),
            "subquery": bool(QA.hasSubquery(root)),
            "numSubqueries": int(QA.countSubqueries(root)),
            "predicates": int(QA.countFilterPredicates(root)),
            "groupByKeys": int(QA.countGroupByKeys(root)),
            "aggs": int(QA.countAggregates(root)),
            "filters": int(QA.countFilters(root)),
            "filterAboveJoin": bool(QA.hasFilterAboveJoin(root)),
            "outerJoin": bool(QA.hasOuterJoin(root)),
            "union": bool(QA.hasUnion(root)),
            "corr": bool(QA.hasCorrelation(root)),
            "distinct": bool(QA.hasDistinct(root)),
            "depth": int(QA.maxDepth(root)),
            "selfJoinSubquery": bool(QA.hasSelfJoinSubquery(root)),
            "hasAggregate": bool(QA.hasAggregate(root)),
            # --- Tier 1: wire existing QueryAnalyzer methods ---
            "hasLimit": bool(QA.hasLimit(root)),
            "hasSort": bool(QA.hasSort(root)),
            "hasCaseWhen": bool(QA.hasCaseWhen(root)),
            "hasLikePattern": bool(QA.hasLikePattern(root)),
            "countAggregateCalls": int(QA.countAggregateCalls(root)),
            "hasInSubquery": bool(QA.hasInSubquery(root)),
            "hasScalarSubquery": bool(QA.hasScalarSubquery(root)),
            "hasExistsSubquery": bool(QA.hasExistsSubquery(root)),
            "hasWindowFunction": bool(QA.hasWindowFunction(root)),
            "countTables": int(QA.countTables(root)),
            "hasSemiJoin": bool(QA.hasSemiJoin(root)),
            "hasNestedSubquery": bool(QA.hasNestedSubquery(root)),
            "countNodes": int(QA.countNodes(root)),
            # --- Tier 2: new QueryAnalyzer methods ---
            "hasGroupByRollup": bool(QA.hasGroupByRollup(root)),
            "countCaseWhen": int(QA.countCaseWhen(root)),
            "hasFunctionInPredicate": bool(QA.hasFunctionInPredicate(root)),
        }
        return features

    except Exception as e:
        logger.warning("Feature extraction failed: %s", e)
        return None
