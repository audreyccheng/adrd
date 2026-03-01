#!/usr/bin/env python3
"""
Run OpenEvolve for Index Selection
===================================
Convenience script to run evolution with proper output directory naming.

Usage:
    python run_openevolve.py --config configs/config_escape_local_optima.yaml --initial initial_programs/initial_program_autoadmin.py
    python run_openevolve.py -c configs/config_workload_aware.yaml -i initial_programs/initial_program.py --iterations 50
    python run_openevolve.py -c configs/config_all_workloads.yaml -i initial_programs/best_explore_extend_1215.py --full

Evaluator options:
    --full: Use evaluator_full.py (all queries: 18+79+33=130, no timeout)
    default: Use evaluator.py (subset: 18+30+10=58 queries, with timeout)
    
Output will be saved to: outputs/{config_name}_{initial_program_name}/
"""

import argparse
import os
import sys
import subprocess
import yaml
from pathlib import Path


def get_name_from_path(filepath):
    """Extract clean name from filepath (without extension and prefix)."""
    name = Path(filepath).stem
    # Remove common prefixes
    for prefix in ["config_", "initial_program_"]:
        if name.startswith(prefix):
            name = name[len(prefix):]
    return name


def main():
    parser = argparse.ArgumentParser(
        description="Run OpenEvolve for Index Selection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_openevolve.py -c configs/config_escape_local_optima.yaml -i initial_programs/initial_program_autoadmin.py
  python run_openevolve.py -c configs/config_workload_aware.yaml -i initial_programs/initial_program.py --iterations 50
  python run_openevolve.py -c configs/config.yaml -i initial_programs/initial_program.py --dry-run
  python run_openevolve.py -c configs/config_all_workloads.yaml -i initial_programs/best_explore_extend_1215.py --full
        """
    )
    
    parser.add_argument(
        "-c", "--config",
        required=True,
        help="Config file (e.g., configs/config_escape_local_optima.yaml)"
    )
    
    parser.add_argument(
        "-i", "--initial",
        required=True,
        help="Initial program file (e.g., initial_programs/initial_program_autoadmin.py)"
    )
    
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Override max_iterations from config"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print command without executing"
    )
    
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override output directory (default: outputs/{config}_{initial}/)"
    )
    
    parser.add_argument(
        "--full",
        action="store_true",
        help="Use evaluator_full.py (all queries, no timeout) instead of evaluator.py (subset, with timeout)"
    )
    
    args = parser.parse_args()
    
    # Validate files exist
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)
    
    if not os.path.exists(args.initial):
        print(f"Error: Initial program not found: {args.initial}")
        sys.exit(1)
    
    # Generate output directory name
    config_name = get_name_from_path(args.config)
    initial_name = get_name_from_path(args.initial)
    
    # Get script directory for absolute paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    if args.output_dir:
        output_dir = os.path.abspath(args.output_dir)
    else:
        # Include timestamp to avoid overwriting previous runs
        from datetime import datetime
        timestamp = datetime.now().strftime("%m%d_%H%M")
        output_dir = os.path.join(script_dir, "outputs", f"{config_name}_{initial_name}_{timestamp}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and modify config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update output_dir in config (use absolute path)
    config['output_dir'] = output_dir
    
    # Override iterations if specified
    if args.iterations:
        config['max_iterations'] = args.iterations
    
    # Write temporary config in script directory (not cwd)
    temp_config = os.path.join(script_dir, f".tmp_config_{config_name}_{initial_name}.yaml")
    with open(temp_config, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    # Build command - use absolute paths
    openevolve_script = os.path.join(script_dir, "deps", "openevolve", "openevolve-run.py")
    initial_abs = os.path.abspath(args.initial)
    
    # Select evaluator based on --full flag
    evaluator_file = "evaluator_full.py" if args.full else "evaluator.py"
    evaluator_abs = os.path.join(script_dir, evaluator_file)
    
    cmd = [
        sys.executable,
        openevolve_script,
        initial_abs,
        evaluator_abs,
        "--config", temp_config,
        "--output", output_dir,
    ]
    
    if args.iterations:
        cmd.extend(["--iterations", str(args.iterations)])
    
    print("=" * 60)
    print("OpenEvolve Index Selection Runner")
    print("=" * 60)
    print(f"Config:          {args.config} ({config_name})")
    print(f"Initial Program: {args.initial} ({initial_name})")
    print(f"Evaluator:       {evaluator_file} ({'FULL workload' if args.full else 'subset workload'})")
    print(f"Output Dir:      {output_dir}")
    print(f"Temp Config:     {temp_config}")
    print(f"Iterations:      {config.get('max_iterations', 'default')}")
    
    # Set benchmark environment variable if specified in config
    env = os.environ.copy()
    # Check top-level eval_benchmark first, then fall back to evaluator.benchmark
    if 'eval_benchmark' in config:
        benchmark = config['eval_benchmark']
        env['EVAL_BENCHMARK'] = benchmark
        print(f"Benchmark:       {benchmark.upper()}")
    elif 'evaluator' in config and 'benchmark' in config['evaluator']:
        benchmark = config['evaluator']['benchmark']
        env['EVAL_BENCHMARK'] = benchmark
        print(f"Benchmark:       {benchmark.upper()}")
    else:
        env['EVAL_BENCHMARK'] = 'tpch'
        print(f"Benchmark:       TPCH (default)")
    print("=" * 60)
    
    if args.dry_run:
        print("\n[DRY RUN] Would execute:")
        print(" ".join(cmd))
        print(f"\nTemp config written to: {temp_config}")
    else:
        print(f"\nStarting evolution...")
        print(f"Command: {' '.join(cmd)}\n")
        
        try:
            # Run the command with updated environment
            result = subprocess.run(cmd, cwd=os.getcwd(), env=env)
            
            # Cleanup temp config
            if os.path.exists(temp_config):
                os.remove(temp_config)
            
            sys.exit(result.returncode)
            
        except KeyboardInterrupt:
            print("\n\nInterrupted by user.")
            if os.path.exists(temp_config):
                os.remove(temp_config)
            sys.exit(1)
        except Exception as e:
            print(f"\nError: {e}")
            if os.path.exists(temp_config):
                os.remove(temp_config)
            sys.exit(1)


if __name__ == "__main__":
    main()

