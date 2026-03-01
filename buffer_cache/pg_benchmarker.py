"""
PGBenchmarker — Automated PostgreSQL benchmarking.

Wraps the existing BenchBase infrastructure to run TPC-H/TPC-C benchmarks
against postgres-pbm with an evolved eviction policy.

Handles: PostgreSQL lifecycle (stop/start), BenchBase execution,
result parsing, and stat collection.
"""

import glob
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Optional

try:
    from .results import BenchmarkResult
except ImportError:
    from results import BenchmarkResult


class PGBenchmarker:
    """
    Benchmarks a postgres-pbm build using BenchBase.

    Configuration matches EXPS_SO_FAR.md defaults:
    - TPC-H: SF=10, 3200MB shared_buffers, 16 terminals, Q1-Q16
    - 10 PBM samples, use_freq=on
    """

    def __init__(
        self,
        pbm_root: Optional[str] = None,
        pgport: int = 5433,
        pgdb: str = "benchbase_tpch",
    ):
        self.pbm_root = pbm_root or os.environ.get("PBM_ROOT", os.path.expanduser("~/pbm-exp"))
        self.pgdata = os.path.join(self.pbm_root, "pgdata-exp")
        self.pgbin = os.path.join(self.pbm_root, "install", "bin")
        self.benchbase = os.path.join(
            self.pbm_root, "postgres", "BenchBase-pbm", "target", "benchbase-postgres"
        )
        self.pgport = pgport
        self.pgdb = pgdb
        self.pguser = os.environ.get("PGUSER", os.environ.get("USER", "postgres"))

    def benchmark_tpch(
        self,
        shared_buffers_mb: int = 3200,
        terminals: int = 16,
        num_samples: int = 10,
        results_dir: Optional[str] = None,
    ) -> BenchmarkResult:
        """
        Run a TPC-H benchmark and return results.

        Args:
            shared_buffers_mb: PostgreSQL shared_buffers size in MB
            terminals: Number of concurrent terminals
            num_samples: PBM eviction sample count
            results_dir: Directory for raw BenchBase output

        Returns:
            BenchmarkResult with throughput, hit_rate, disk_reads, runtime
        """
        if results_dir is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            results_dir = os.path.join(self.pbm_root, "exp-results", f"simevolver_{timestamp}")
        os.makedirs(results_dir, exist_ok=True)

        try:
            # 1. Stop any running PostgreSQL
            self._pg_stop()

            # 2. Clear OS page cache
            self._clear_cache()

            # 3. Start PostgreSQL with configured settings
            self._pg_start(shared_buffers_mb, num_samples)

            # 4. Reset stats
            self._psql("SELECT pg_stat_reset();")

            # 5. Create TPC-H config
            config_path = self._create_tpch_config(terminals)

            # 6. Run BenchBase
            start_time = time.time()
            self._run_benchbase(config_path, results_dir)
            runtime = time.time() - start_time

            # 7. Collect results
            throughput = self._parse_throughput(results_dir)
            hit_rate, disk_reads = self._collect_pg_stats()

            # 8. Stop PostgreSQL
            self._pg_stop()

            return BenchmarkResult(
                throughput=throughput,
                hit_rate=hit_rate / 100.0,  # Convert percentage to 0-1
                disk_reads=disk_reads,
                runtime_seconds=runtime,
                benchmark_type="tpch",
                config_details={
                    "shared_buffers_mb": shared_buffers_mb,
                    "terminals": terminals,
                    "num_samples": num_samples,
                    "scale_factor": 10,
                },
            )

        except Exception as e:
            print(f"Benchmark failed: {e}")
            self._pg_stop()
            return BenchmarkResult(
                throughput=0.0,
                hit_rate=0.0,
                disk_reads=0,
                runtime_seconds=0.0,
                benchmark_type="tpch",
                config_details={"error": str(e)},
            )

    def _pg_stop(self):
        """Stop PostgreSQL if running."""
        subprocess.run(
            [os.path.join(self.pgbin, "pg_ctl"), "-D", self.pgdata, "stop", "-m", "fast"],
            capture_output=True,
        )
        time.sleep(2)

    def _pg_start(self, buffer_mb: int, samples: int):
        """Start PostgreSQL with specified settings."""
        logfile = os.path.join(self.pgdata, "logfile")
        result = subprocess.run(
            [
                os.path.join(self.pgbin, "pg_ctl"),
                "-D", self.pgdata,
                "-l", logfile,
                "start",
                "-o", (
                    f"-p {self.pgport} "
                    f"-c shared_buffers={buffer_mb}MB "
                    "-c synchronize_seqscans=on "
                    "-c track_io_timing=on "
                    "-c random_page_cost=1.1 "
                    f"-c pbm_evict_num_samples={samples} "
                    "-c pbm_evict_use_freq=on "
                    "-c max_connections=200 "
                    "-c max_locks_per_transaction=256 "
                    "-c max_pred_locks_per_transaction=256 "
                    "-c log_min_messages=warning "
                    "-c autovacuum=off"
                ),
            ],
            capture_output=True,
            text=True,
        )
        time.sleep(3)

        # Verify PostgreSQL started
        check = subprocess.run(
            [os.path.join(self.pgbin, "pg_ctl"), "-D", self.pgdata, "status"],
            capture_output=True,
        )
        if check.returncode != 0:
            raise RuntimeError("PostgreSQL failed to start")

    def _clear_cache(self):
        """Clear OS page cache."""
        try:
            if os.path.exists("/proc/sys/vm/drop_caches"):
                subprocess.run(
                    ["sudo", "sh", "-c", "sync; echo 3 > /proc/sys/vm/drop_caches"],
                    capture_output=True,
                )
            else:
                subprocess.run(["sudo", "purge"], capture_output=True)
        except Exception:
            print("WARNING: Could not clear OS cache")

    def _psql(self, query: str) -> str:
        """Run a psql query and return output."""
        result = subprocess.run(
            [
                os.path.join(self.pgbin, "psql"),
                "-p", str(self.pgport),
                "-d", self.pgdb,
                "-t", "-A",
                "-c", query,
            ],
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    def _create_tpch_config(self, terminals: int) -> str:
        """Create BenchBase TPC-H config XML."""
        config_path = "/tmp/simevolver_tpch.xml"
        config = f"""<?xml version="1.0"?>
<parameters>
    <type>POSTGRES</type>
    <driver>org.postgresql.Driver</driver>
    <url>jdbc:postgresql://localhost:{self.pgport}/{self.pgdb}?sslmode=disable</url>
    <username>{self.pguser}</username>
    <password></password>
    <isolation>TRANSACTION_READ_COMMITTED</isolation>
    <scalefactor>10</scalefactor>
    <terminals>{terminals}</terminals>
    <works>
        <work>
            <rate>unlimited</rate>
            <counts>1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0</counts>
        </work>
    </works>
    <transactiontypes>
        <transactiontype><name>Q1</name><id>1</id></transactiontype>
        <transactiontype><name>Q2</name><id>2</id></transactiontype>
        <transactiontype><name>Q3</name><id>3</id></transactiontype>
        <transactiontype><name>Q4</name><id>4</id></transactiontype>
        <transactiontype><name>Q5</name><id>5</id></transactiontype>
        <transactiontype><name>Q6</name><id>6</id></transactiontype>
        <transactiontype><name>Q7</name><id>7</id></transactiontype>
        <transactiontype><name>Q8</name><id>8</id></transactiontype>
        <transactiontype><name>Q9</name><id>9</id></transactiontype>
        <transactiontype><name>Q10</name><id>10</id></transactiontype>
        <transactiontype><name>Q11</name><id>11</id></transactiontype>
        <transactiontype><name>Q12</name><id>12</id></transactiontype>
        <transactiontype><name>Q13</name><id>13</id></transactiontype>
        <transactiontype><name>Q14</name><id>14</id></transactiontype>
        <transactiontype><name>Q15</name><id>15</id></transactiontype>
        <transactiontype><name>Q16</name><id>16</id></transactiontype>
        <transactiontype><name>Q17</name><id>17</id></transactiontype>
        <transactiontype><name>Q18</name><id>18</id></transactiontype>
        <transactiontype><name>Q19</name><id>19</id></transactiontype>
        <transactiontype><name>Q20</name><id>20</id></transactiontype>
        <transactiontype><name>Q21</name><id>21</id></transactiontype>
        <transactiontype><name>Q22</name><id>22</id></transactiontype>
    </transactiontypes>
</parameters>"""
        with open(config_path, "w") as f:
            f.write(config)
        return config_path

    def _run_benchbase(self, config_path: str, results_dir: str):
        """Run BenchBase TPC-H workload."""
        jar_path = os.path.join(self.benchbase, "benchbase.jar")
        if not os.path.exists(jar_path):
            raise FileNotFoundError(f"BenchBase jar not found at {jar_path}")

        result = subprocess.run(
            [
                "java", "-jar", jar_path,
                "-b", "tpch",
                "-c", config_path,
                "--create=false", "--load=false", "--execute=true",
                "-d", results_dir,
            ],
            cwd=self.benchbase,
            capture_output=True,
            text=True,
            timeout=1800,  # 30 minute timeout
        )
        if result.returncode != 0:
            raise RuntimeError(f"BenchBase failed: {result.stderr[-500:]}")

    def _parse_throughput(self, results_dir: str) -> float:
        """Parse throughput from BenchBase summary JSON."""
        summary_files = glob.glob(os.path.join(results_dir, "*.summary.json"))
        if not summary_files:
            return 0.0

        with open(summary_files[0]) as f:
            data = json.load(f)

        # BenchBase stores throughput as "Throughput (requests/second)"
        return float(data.get("Throughput (requests/second)", 0.0))

    def _collect_pg_stats(self) -> tuple:
        """Collect buffer cache statistics from PostgreSQL."""
        # Hit rate
        hit_rate_str = self._psql(
            "SELECT ROUND(100.0 * SUM(heap_blks_hit + idx_blks_hit) / "
            "NULLIF(SUM(heap_blks_hit + heap_blks_read + idx_blks_hit + idx_blks_read), 0), 2) "
            "FROM pg_statio_user_tables;"
        )
        hit_rate = float(hit_rate_str) if hit_rate_str else 0.0

        # Disk reads
        disk_reads_str = self._psql(
            "SELECT SUM(heap_blks_read + idx_blks_read) FROM pg_statio_user_tables;"
        )
        disk_reads = int(disk_reads_str) if disk_reads_str else 0

        return hit_rate, disk_reads
