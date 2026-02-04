#!/usr/bin/env python3
"""
批量运行多配置实验

用于系统性地测试不同 LLM 模型、任务数量、抽样策略等配置下的 benchmark 结果。
结果会以类似 AgentBusters-Leaderboard 的格式保存，便于对比分析。

Usage:
    # 运行特定实验
    python scripts/run_experiments.py --experiment model_comparison
    
    # 运行所有实验
    python scripts/run_experiments.py --all
    
    # 只运行实验中的特定配置
    python scripts/run_experiments.py --experiment model_comparison --config-id llama3.1-70b
    
    # 指定输出目录
    python scripts/run_experiments.py --experiment scale_comparison --output-dir results/scale_test
    
    # 列出所有可用实验
    python scripts/run_experiments.py --list

Examples:
    # 快速测试：只运行一个小规模配置
    python scripts/run_experiments.py --experiment scale_comparison --config-id scale-10
    
    # 完整模型对比
    python scripts/run_experiments.py --experiment model_comparison
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    import yaml
except ImportError:
    print("Error: pyyaml required. Install with: pip install pyyaml")
    sys.exit(1)


def load_experiment_configs(config_path: str) -> dict:
    """加载实验配置文件"""
    path = Path(config_path)
    if not path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
    
    with open(path) as f:
        return yaml.safe_load(f)


def update_env_file(
    env_path: str,
    llm_model: str,
    llm_api_base: Optional[str] = None,
    llm_temperature: float = 0.0,
    llm_provider: str = "openai",
) -> dict:
    """更新 .env 文件中的 LLM 配置并返回原始值"""
    from dotenv import dotenv_values, set_key
    
    path = Path(env_path)
    original = dotenv_values(path) if path.exists() else {}
    
    # 设置新值
    set_key(path, "LLM_MODEL", llm_model)
    set_key(path, "PURPLE_LLM_TEMPERATURE", str(llm_temperature))
    set_key(path, "LLM_PROVIDER", llm_provider)
    
    if llm_api_base:
        set_key(path, "OPENAI_API_BASE", llm_api_base)
        set_key(path, "OPENAI_BASE_URL", llm_api_base)
    
    return original


def run_single_experiment(
    config_id: str,
    llm_model: str,
    eval_config: str,
    num_tasks: int,
    output_dir: str,
    llm_api_base: Optional[str] = None,
    llm_temperature: float = 0.0,
    llm_provider: str = "openai",
    green_url: str = "http://localhost:9109",
    purple_url: str = "http://localhost:9110",
    timeout: int = 3600,
    dry_run: bool = False,
) -> dict:
    """运行单个实验配置"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = Path(output_dir) / f"{config_id}_{timestamp}.json"
    
    print(f"\n{'='*70}")
    print(f"🚀 Running experiment: {config_id}")
    print(f"{'='*70}")
    print(f"  Model:      {llm_model}")
    print(f"  API Base:   {llm_api_base or 'default'}")
    print(f"  Config:     {eval_config}")
    print(f"  Tasks:      {num_tasks}")
    print(f"  Timeout:    {timeout}s")
    print(f"  Output:     {output_file}")
    print()
    
    if dry_run:
        print("  [DRY RUN - skipping actual execution]")
        return {
            "config_id": config_id,
            "llm_model": llm_model,
            "num_tasks": num_tasks,
            "output_file": str(output_file),
            "success": True,
            "dry_run": True,
            "timestamp": timestamp,
        }
    
    # 设置环境变量
    env = os.environ.copy()
    env["LLM_MODEL"] = llm_model
    env["PURPLE_LLM_TEMPERATURE"] = str(llm_temperature)
    env["LLM_PROVIDER"] = llm_provider
    
    if llm_api_base:
        env["OPENAI_API_BASE"] = llm_api_base
        env["OPENAI_BASE_URL"] = llm_api_base
    
    # 构建命令
    cmd = [
        sys.executable,
        "scripts/run_a2a_eval.py",
        "--green-url", green_url,
        "--purple-url", purple_url,
        "--num-tasks", str(num_tasks),
        "--timeout", str(timeout),
        "-v",
        "-o", str(output_file),
    ]
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=False,
            cwd=Path(__file__).parent.parent,
        )
        success = result.returncode == 0
    except Exception as e:
        print(f"  ❌ Error: {e}")
        success = False
    
    elapsed = time.time() - start_time
    
    result_info = {
        "config_id": config_id,
        "llm_model": llm_model,
        "llm_api_base": llm_api_base,
        "llm_temperature": llm_temperature,
        "eval_config": eval_config,
        "num_tasks": num_tasks,
        "output_file": str(output_file),
        "success": success,
        "elapsed_seconds": round(elapsed, 2),
        "timestamp": timestamp,
    }
    
    if success:
        print(f"  ✅ Completed in {elapsed/60:.1f} minutes")
    else:
        print(f"  ❌ Failed after {elapsed/60:.1f} minutes")
    
    return result_info


def run_experiment_suite(
    experiment_name: str,
    configs: list,
    output_dir: str,
    config_id_filter: Optional[str] = None,
    dry_run: bool = False,
    **kwargs,
) -> list:
    """运行一组实验"""
    
    results = []
    total = len(configs)
    
    for i, config in enumerate(configs, 1):
        cid = config["id"]
        
        # 如果指定了特定配置，只运行匹配的
        if config_id_filter and cid != config_id_filter:
            continue
        
        print(f"\n[{i}/{total}] {experiment_name} / {cid}")
        
        result = run_single_experiment(
            config_id=cid,
            llm_model=config.get("llm_model", os.getenv("LLM_MODEL", "gpt-4o")),
            eval_config=config.get("eval_config", "config/eval_config.yaml"),
            num_tasks=config.get("num_tasks", 100),
            output_dir=output_dir,
            llm_api_base=config.get("llm_api_base"),
            llm_temperature=config.get("llm_temperature", 0.0),
            llm_provider=config.get("llm_provider", "openai"),
            dry_run=dry_run,
            **kwargs,
        )
        results.append(result)
        
        # 保存中间结果
        summary_file = Path(output_dir) / f"{experiment_name}_progress.json"
        with open(summary_file, "w") as f:
            json.dump(results, f, indent=2)
    
    return results


def list_experiments(config: dict):
    """列出所有可用的实验"""
    print("\n" + "="*70)
    print("Available Experiments")
    print("="*70)
    
    for exp in config.get("experiments", []):
        print(f"\n📋 {exp['name']}")
        print(f"   {exp['description']}")
        print("   Configurations:")
        for cfg in exp.get("configs", []):
            model = cfg.get("llm_model", "default")
            tasks = cfg.get("num_tasks", "?")
            print(f"     - {cfg['id']}: {model} ({tasks} tasks)")


def main():
    parser = argparse.ArgumentParser(
        description="Run multiple experiment configurations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config",
        default="experiments/experiment_configs.yaml",
        help="Path to experiment configs YAML",
    )
    parser.add_argument(
        "--experiment",
        help="Specific experiment to run (e.g., model_comparison)",
    )
    parser.add_argument(
        "--config-id",
        help="Only run specific config within experiment",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all experiments",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available experiments and exit",
    )
    parser.add_argument(
        "--output-dir",
        default="results/experiments",
        help="Directory to save results",
    )
    parser.add_argument(
        "--green-url",
        default="http://localhost:9109",
        help="Green Agent URL",
    )
    parser.add_argument(
        "--purple-url",
        default="http://localhost:9110",
        help="Purple Agent URL",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="Timeout per experiment in seconds",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be run without executing",
    )
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_experiment_configs(args.config)
    
    # 列出实验
    if args.list:
        list_experiments(config)
        return
    
    # 验证参数
    if not args.all and not args.experiment:
        print("Error: Specify --experiment <name> or --all")
        print("Use --list to see available experiments")
        sys.exit(1)
    
    # 创建输出目录
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    all_results = []
    experiments_run = 0
    
    for experiment in config.get("experiments", []):
        if args.all or args.experiment == experiment["name"]:
            print(f"\n{'#'*70}")
            print(f"# Experiment: {experiment['name']}")
            print(f"# {experiment['description']}")
            print(f"{'#'*70}")
            
            results = run_experiment_suite(
                experiment["name"],
                experiment["configs"],
                args.output_dir,
                config_id_filter=args.config_id,
                dry_run=args.dry_run,
                green_url=args.green_url,
                purple_url=args.purple_url,
                timeout=args.timeout,
            )
            all_results.extend(results)
            experiments_run += 1
    
    if experiments_run == 0:
        print(f"Error: Experiment '{args.experiment}' not found")
        print("Use --list to see available experiments")
        sys.exit(1)
    
    # 保存最终汇总
    summary_file = Path(args.output_dir) / "experiment_summary.json"
    with open(summary_file, "w") as f:
        json.dump({
            "generated_at": datetime.now().isoformat(),
            "experiments_run": experiments_run,
            "total_configs": len(all_results),
            "successful": sum(1 for r in all_results if r.get("success")),
            "results": all_results,
        }, f, indent=2)
    
    # 打印摘要
    print(f"\n{'='*70}")
    print("EXPERIMENT SUMMARY")
    print("="*70)
    
    successful = sum(1 for r in all_results if r.get("success"))
    failed = len(all_results) - successful
    
    print(f"  Total configs run: {len(all_results)}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"\n✅ Summary saved to: {summary_file}")
    
    if failed > 0:
        print("\n⚠️  Some experiments failed. Check individual logs for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
