"""
PolicyTranslator — LLM-assisted translation of Python policies to C.

Translates the best evolved Python policy from the inner loop into C code
for integration into postgres-pbm's buffer eviction system (pbm.c).

Only invoked when the inner loop produces a new best policy that exceeds
the previous best by a configurable threshold.
"""

import os
import re
import subprocess
import textwrap
from pathlib import Path
from typing import Optional

# Path to postgres-pbm source.
# Override with SIMEVOLVER_PBM_DIR env var, or clone/submodule into pg_clean/postgres-pbm/
POSTGRES_PBM_DIR = os.environ.get(
    "SIMEVOLVER_PBM_DIR",
    str(Path(__file__).parent / "postgres-pbm"),
)
PBM_C_PATH = os.path.join(POSTGRES_PBM_DIR, "src", "backend", "storage", "buffer", "pbm.c")
PBM_CONFIG_H = os.path.join(POSTGRES_PBM_DIR, "src", "include", "storage", "pbm", "pbm_config.h")


# Template for the translation prompt sent to the LLM
TRANSLATION_PROMPT = textwrap.dedent("""\
You are translating a Python buffer eviction policy into C for PostgreSQL's
buffer manager (postgres-pbm).

## Python Policy to Translate
```python
{python_code}
```

## Target C Integration Point
The translated code should implement a `compute_eviction_score_evolved()` function
that returns a score for a given buffer. Higher score = evict sooner.

## Available C Data Structures
- `BufferDesc *buf`: Buffer descriptor
  - `buf->buf_id`: Buffer ID
  - `buf->tag`: BufferTag (reltablespace, reldatabase, relNumber, blockNum, forkNum)
  - `GetRefCount(buf)` / `GetUsageCount(buf)`: Atomic reference/usage counts
  - `buf->state`: Packed state (use BUF_STATE_GET_REFCOUNT/USAGECOUNT macros)
- `BlockGroupData *bg`: Block group (128 blocks = 1MiB)
  - `bg->scans_list`: Linked list of active scans
  - `bg->n_active_scans`: Atomic scan count
  - `BlockGroupTimeToNextConsumption(bg)`: Predicted next access time
- `PbmBufferMeta *meta`: Per-buffer PBM metadata
  - `pg_atomic_read_u32(&meta->nrecent_accesses)`: Recent access count
  - `pg_atomic_read_u64(&meta->last_access)`: Last access timestamp

## Example: Existing compute_eviction_score_v2() (COMBINED_V2)
```c
static double
compute_eviction_score_v2(BufferDesc *buf, PbmBufferMeta *meta, BlockGroupData *bg)
{{
    double score = 0.0;
    int usage_count = GetUsageCount(buf);
    int naccesses = pg_atomic_read_u32(&meta->nrecent_accesses);
    uint64 last_access = pg_atomic_read_u64(&meta->last_access);
    uint64 now = GetPbmCurrentTick();
    double ms_since_access = PbmTicksToMs(now - last_access);

    // Base: PBM prediction (time to next scan access)
    double next_access_ms = BlockGroupTimeToNextConsumption(bg);
    score = next_access_ms;

    // Multi-scan protection
    int n_scans = pg_atomic_read_u32(&bg->n_active_scans);
    if (n_scans > 1) score -= 10.0 * n_scans;

    // Clean page preference
    if (!(buf->state & BM_DIRTY)) score += 5.0;

    // Cold page preference
    if (usage_count <= 1) score += 25.0;

    // Frequency protection
    if (naccesses > 0 && ms_since_access > 0) {{
        double est_inter_access = ms_since_access / naccesses;
        if (est_inter_access < 1000.0) score -= 20.0;
    }}

    return score;
}}
```

## Requirements
1. Return a single C function: `static double compute_eviction_score_evolved(...)`
2. Use only the data structures listed above
3. No memory allocation (no malloc)
4. No locks (use atomic reads for meta fields)
5. Keep it simple — minimize branches and function calls
6. Translate the Python logic faithfully, adapting to C idioms

Return ONLY the C function, wrapped in ```c code fences.
""")


class PolicyTranslator:
    """
    Translates Python policies to C for postgres-pbm integration.
    Uses an LLM API (OpenAI-compatible) for translation.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o"):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model
        self.last_translation: Optional[str] = None
        self.last_python_source: Optional[str] = None

    def translate(self, python_policy_path: str) -> Optional[str]:
        """
        Translate a Python policy file to C.

        Args:
            python_policy_path: Path to the Python policy file

        Returns:
            C source code string, or None on failure
        """
        with open(python_policy_path) as f:
            python_code = f.read()

        self.last_python_source = python_code
        prompt = TRANSLATION_PROMPT.format(python_code=python_code)

        try:
            c_code = self._call_llm(prompt)
            if c_code:
                # Extract code from fences
                match = re.search(r'```c\s*(.*?)```', c_code, re.DOTALL)
                if match:
                    c_code = match.group(1).strip()
                self.last_translation = c_code
                return c_code
        except Exception as e:
            print(f"Translation failed: {e}")
            return None

    def _call_llm(self, prompt: str) -> Optional[str]:
        """Call the LLM API for translation."""
        if not self.api_key:
            print("WARNING: No API key set. Returning placeholder translation.")
            return self._placeholder_translation()

        try:
            import openai
            client = openai.OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert C programmer specializing in PostgreSQL internals."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=4000,
            )
            return response.choices[0].message.content
        except ImportError:
            print("openai package not installed. Using placeholder.")
            return self._placeholder_translation()
        except Exception as e:
            print(f"LLM API call failed: {e}")
            return None

    def _placeholder_translation(self) -> str:
        """Return a placeholder C function for testing without an API key."""
        return textwrap.dedent("""\
            ```c
            static double
            compute_eviction_score_evolved(BufferDesc *buf, PbmBufferMeta *meta, BlockGroupData *bg)
            {
                /* Placeholder — replace with LLM-translated policy */
                double score = 0.0;
                int usage_count = GetUsageCount(buf);

                /* Base: PBM prediction */
                double next_access_ms = BlockGroupTimeToNextConsumption(bg);
                score = next_access_ms;

                /* Clean page preference */
                if (!(buf->state & BM_DIRTY))
                    score += 5.0;

                /* Cold page preference */
                if (usage_count <= 1)
                    score += 25.0;

                return score;
            }
            ```
        """)

    def integrate(self, c_code: str, mode_name: str = "EVOLVED") -> bool:
        """
        Integrate translated C code into pbm.c as a new eviction mode.

        This writes the function to a separate file that can be #included,
        avoiding direct modification of pbm.c (which is complex and risky).

        Args:
            c_code: The translated C function
            mode_name: Name for the eviction mode

        Returns:
            True if integration file was written successfully
        """
        output_path = os.path.join(
            POSTGRES_PBM_DIR, "src", "backend", "storage", "buffer",
            "evolved_score.h"
        )

        header = textwrap.dedent(f"""\
            /*
             * Auto-generated by sim_evolver PolicyTranslator.
             * Evolved eviction scoring function for mode: {mode_name}
             *
             * DO NOT EDIT — regenerated on each outer loop iteration.
             */

            #ifndef EVOLVED_SCORE_H
            #define EVOLVED_SCORE_H

        """)

        footer = "\n\n#endif /* EVOLVED_SCORE_H */\n"

        with open(output_path, "w") as f:
            f.write(header)
            f.write(c_code)
            f.write(footer)

        print(f"Wrote evolved scoring function to {output_path}")
        return True

    def compile(self) -> bool:
        """
        Build postgres-pbm with the integrated evolved scoring function.

        Returns:
            True if compilation succeeded
        """
        try:
            result = subprocess.run(
                ["make", "-j8"],
                cwd=POSTGRES_PBM_DIR,
                capture_output=True,
                text=True,
                timeout=300,
            )
            if result.returncode != 0:
                print(f"Compilation failed:\n{result.stderr[-1000:]}")
                return False

            result = subprocess.run(
                ["make", "install"],
                cwd=POSTGRES_PBM_DIR,
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode != 0:
                print(f"Install failed:\n{result.stderr[-500:]}")
                return False

            print("PostgreSQL build and install succeeded")
            return True

        except subprocess.TimeoutExpired:
            print("Build timed out")
            return False
        except Exception as e:
            print(f"Build error: {e}")
            return False

    def save_translation(self, output_dir: str, config_name: str):
        """Save the translation artifacts for reproducibility."""
        path = Path(output_dir) / "translations"
        path.mkdir(parents=True, exist_ok=True)

        if self.last_python_source:
            with open(path / f"{config_name}_policy.py", "w") as f:
                f.write(self.last_python_source)

        if self.last_translation:
            with open(path / f"{config_name}_score.c", "w") as f:
                f.write(self.last_translation)
