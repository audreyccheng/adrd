"""
PostgreSQL query latency measurement utilities.

Provides:
- get_connection(): Get a psycopg2 connection for a benchmark
- measure_latency(): Measure single query execution time
- measure_latency_robust(): Multi-run measurement with median
"""

import re
import time
import logging
import statistics
from typing import Dict, Optional, Tuple

import psycopg2
import psycopg2.errors

logger = logging.getLogger(__name__)


def get_connection(pg_config: Dict[str, str]):
    """Get a PostgreSQL connection from config dict.

    Args:
        pg_config: Dict with keys: host, port, dbname, user, password

    Returns:
        psycopg2 connection object
    """
    return psycopg2.connect(
        host=pg_config["host"],
        port=int(pg_config["port"]),
        dbname=pg_config["dbname"],
        user=pg_config["user"],
        password=pg_config.get("password", ""),
    )


def _clean_sql(sql: str) -> str:
    """Clean Calcite-generated SQL for PostgreSQL execution."""
    clean = re.sub(r'"CATALOG"\."SALES"\.', "", sql)
    clean = re.sub(r'FETCH NEXT (\d+) ROWS ONLY', r'LIMIT \1', clean)
    clean = " ".join(clean.split())
    return clean


def measure_latency(
    conn,
    sql: str,
    timeout_sec: int = 120,
    disable_indexscan: bool = False,
) -> Tuple[Optional[float], str]:
    """Measure single query execution latency.

    Args:
        conn: psycopg2 connection (autocommit recommended)
        sql: SQL query to execute
        timeout_sec: Query timeout in seconds
        disable_indexscan: If True, SET enable_indexscan = off (matches R-Bot paper)

    Returns:
        (latency_seconds, status) where status is "OK", "TIMEOUT", or "ERROR: ..."
    """
    clean = _clean_sql(sql)

    try:
        conn.autocommit = True
        cursor = conn.cursor()
        cursor.execute(f"SET statement_timeout = '{timeout_sec}s'")
        if disable_indexscan:
            cursor.execute("SET enable_indexscan = off")

        start = time.time()
        cursor.execute(clean)
        cursor.fetchall()
        latency = time.time() - start
        cursor.close()
        return latency, "OK"
    except psycopg2.errors.QueryCanceled:
        conn.rollback()
        return None, "TIMEOUT"
    except Exception as e:
        try:
            conn.rollback()
        except Exception:
            pass
        return None, f"ERROR: {str(e)[:100]}"


def measure_latency_robust(
    conn,
    sql: str,
    runs: int = 5,
    warmup: int = 1,
    timeout_sec: int = 120,
    max_failures: int = 1,
    disable_indexscan: bool = False,
) -> Tuple[Optional[float], str]:
    """Multi-run latency measurement with warmup and median.

    Args:
        conn: psycopg2 connection
        sql: SQL query to execute
        runs: Number of timed runs
        warmup: Number of warmup runs (not included in measurement)
        timeout_sec: Per-run timeout in seconds
        max_failures: Max transient failures to tolerate during timed runs
        disable_indexscan: If True, SET enable_indexscan = off (matches R-Bot paper)

    Returns:
        (median_latency_seconds, status) where status is "OK", "TIMEOUT", or "ERROR: ..."
    """
    # Warmup runs — fail-fast (if warmup fails, query is genuinely broken)
    for _ in range(warmup):
        lat, status = measure_latency(conn, sql, timeout_sec, disable_indexscan)
        if status != "OK":
            return lat, status

    # Timed runs — tolerate up to max_failures transient failures
    latencies = []
    failures = 0
    last_failure_status = "ERROR: all runs failed"
    for _ in range(runs):
        lat, status = measure_latency(conn, sql, timeout_sec, disable_indexscan)
        if status != "OK":
            failures += 1
            last_failure_status = status
            if failures > max_failures:
                return lat, status
            continue
        latencies.append(lat)

    if not latencies:
        return None, last_failure_status

    median_lat = statistics.median(latencies)
    return median_lat, "OK"
