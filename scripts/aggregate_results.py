#!/usr/bin/env python3
"""
汇总和分析多次实验结果

生成类似 Leaderboard 的结果表格，支持多种输出格式。

Usage:
    python scripts/aggregate_results.py results/experiments
    python scripts/aggregate_results.py results/experiments --format markdown
    python scripts/aggregate_results.py results/experiments --sort-by Overall --top 10
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


def load_result_file(file_path: Path) -> Optional[dict]:
    """加载单个结果文件"""
    try:
        with open(file_path) as f:
            data = json.load(f)
        return data
    except (json.JSONDecodeError, IOError) as e:
        print(f"Warning: Failed to load {file_path}: {e}")
        return None


def extract_scores(data: dict) -> dict:
    """从结果数据中提取分数"""
    scores = {
        "overall": 0,
        "knowledge": 0,
        "analysis": 0,
        "options": 0,
        "crypto": 0,
        "gdpval": 0,
    }
    
    # 尝试不同的结果格式
    if "result" in data:
        result = data["result"]
        if isinstance(result, dict):
            task = result.get("task", result)
            artifacts = task.get("artifacts", [])
            
            for artifact in artifacts:
                for part in artifact.get("parts", []):
                    root = part.get("root", part)
                    if root.get("kind") == "data" or "data" in root:
                        eval_data = root.get("data", {})
                        
                        # 提取分数
                        if "overall_score" in eval_data:
                            scores["overall"] = eval_data["overall_score"]
                        if "section_scores" in eval_data:
                            ss = eval_data["section_scores"]
                            scores["knowledge"] = ss.get("knowledge", 0)
                            scores["analysis"] = ss.get("analysis", 0)
                            scores["options"] = ss.get("options", 0)
                            scores["crypto"] = ss.get("crypto", 0)
                        if "dataset_scores" in eval_data:
                            ds = eval_data["dataset_scores"]
                            scores["gdpval"] = ds.get("gdpval", 0) * 100
    
    return scores


def load_all_results(results_dir: str) -> list:
    """加载目录中的所有结果文件"""
    results = []
    dir_path = Path(results_dir)
    
    if not dir_path.exists():
        print(f"Error: Directory not found: {results_dir}")
        sys.exit(1)
    
    for file_path in dir_path.glob("*.json"):
        # 跳过汇总文件
        if file_path.name in ["experiment_summary.json", "leaderboard.csv"]:
            continue
        if "_progress.json" in file_path.name:
            continue
        
        data = load_result_file(file_path)
        if data:
            data["_filename"] = file_path.name
            results.append(data)
    
    print(f"Loaded {len(results)} result files from {results_dir}")
    return results


def create_leaderboard(results: list) -> list:
    """创建排行榜数据"""
    rows = []
    
    for r in results:
        filename = r.get("_filename", "unknown")
        
        # 尝试从文件名解析配置 ID
        config_id = filename.replace(".json", "").rsplit("_", 2)[0] if "_" in filename else filename
        
        # 从结果中提取配置信息
        config = {}
        if "config" in r:
            config = r["config"]
        
        # 提取分数
        scores = extract_scores(r)
        
        # 从 experiment_summary 或其他来源获取模型信息
        llm_model = config.get("llm_model", "")
        if not llm_model:
            # 尝试从文件名推断
            if "llama" in filename.lower():
                llm_model = "llama"
            elif "qwen" in filename.lower():
                llm_model = "qwen"
            elif "gpt" in filename.lower():
                llm_model = "gpt"
        
        rows.append({
            "Config ID": config_id,
            "Model": llm_model or "unknown",
            "Tasks": config.get("num_tasks", "?"),
            "Overall": round(scores["overall"], 2),
            "Knowledge": round(scores["knowledge"], 2),
            "Analysis": round(scores["analysis"], 2),
            "Options": round(scores["options"], 2),
            "Crypto": round(scores["crypto"], 2),
            "GDPVal": round(scores["gdpval"], 2),
            "File": filename,
        })
    
    return rows


def format_table_text(rows: list, sort_by: str = "Overall", top_n: Optional[int] = None) -> str:
    """格式化为文本表格"""
    if not rows:
        return "No results to display"
    
    # 排序
    if sort_by in rows[0]:
        rows = sorted(rows, key=lambda x: x.get(sort_by, 0), reverse=True)
    
    # 截取
    if top_n:
        rows = rows[:top_n]
    
    # 计算列宽
    headers = ["Config ID", "Model", "Tasks", "Overall", "Knowledge", "Analysis", "Options", "Crypto", "GDPVal"]
    widths = {h: len(h) for h in headers}
    
    for row in rows:
        for h in headers:
            val = str(row.get(h, ""))
            widths[h] = max(widths[h], len(val))
    
    # 生成表格
    lines = []
    
    # Header
    header_line = " | ".join(h.ljust(widths[h]) for h in headers)
    separator = "-+-".join("-" * widths[h] for h in headers)
    
    lines.append(header_line)
    lines.append(separator)
    
    # Rows
    for row in rows:
        line = " | ".join(str(row.get(h, "")).ljust(widths[h]) for h in headers)
        lines.append(line)
    
    return "\n".join(lines)


def format_table_markdown(rows: list, sort_by: str = "Overall", top_n: Optional[int] = None) -> str:
    """格式化为 Markdown 表格"""
    if not rows:
        return "No results to display"
    
    # 排序
    if sort_by in rows[0]:
        rows = sorted(rows, key=lambda x: x.get(sort_by, 0), reverse=True)
    
    # 截取
    if top_n:
        rows = rows[:top_n]
    
    headers = ["Config ID", "Model", "Tasks", "Overall", "Knowledge", "Analysis", "Options", "Crypto", "GDPVal"]
    
    lines = []
    
    # Header
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    
    # Rows
    for row in rows:
        values = [str(row.get(h, "")) for h in headers]
        lines.append("| " + " | ".join(values) + " |")
    
    return "\n".join(lines)


def format_table_csv(rows: list, sort_by: str = "Overall", top_n: Optional[int] = None) -> str:
    """格式化为 CSV"""
    if not rows:
        return ""
    
    # 排序
    if sort_by in rows[0]:
        rows = sorted(rows, key=lambda x: x.get(sort_by, 0), reverse=True)
    
    # 截取
    if top_n:
        rows = rows[:top_n]
    
    headers = ["Config ID", "Model", "Tasks", "Overall", "Knowledge", "Analysis", "Options", "Crypto", "GDPVal", "File"]
    
    lines = [",".join(headers)]
    for row in rows:
        values = [str(row.get(h, "")).replace(",", ";") for h in headers]
        lines.append(",".join(values))
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate and analyze experiment results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "results_dir",
        help="Directory containing result JSON files",
    )
    parser.add_argument(
        "--format",
        choices=["text", "markdown", "csv", "json"],
        default="text",
        help="Output format (default: text)",
    )
    parser.add_argument(
        "--sort-by",
        default="Overall",
        help="Column to sort by (default: Overall)",
    )
    parser.add_argument(
        "--top",
        type=int,
        help="Only show top N results",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output file (default: stdout)",
    )
    
    args = parser.parse_args()
    
    # 加载结果
    results = load_all_results(args.results_dir)
    
    if not results:
        print("No results found")
        sys.exit(1)
    
    # 创建排行榜
    leaderboard = create_leaderboard(results)
    
    # 格式化
    if args.format == "text":
        output = format_table_text(leaderboard, args.sort_by, args.top)
    elif args.format == "markdown":
        output = format_table_markdown(leaderboard, args.sort_by, args.top)
    elif args.format == "csv":
        output = format_table_csv(leaderboard, args.sort_by, args.top)
    elif args.format == "json":
        # 排序
        if args.sort_by in leaderboard[0]:
            leaderboard = sorted(leaderboard, key=lambda x: x.get(args.sort_by, 0), reverse=True)
        if args.top:
            leaderboard = leaderboard[:args.top]
        output = json.dumps({
            "generated_at": datetime.now().isoformat(),
            "total_results": len(leaderboard),
            "leaderboard": leaderboard,
        }, indent=2)
    
    # 输出
    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
        print(f"✅ Results saved to: {args.output}")
    else:
        print("\n" + "="*80)
        print("BENCHMARK LEADERBOARD")
        print("="*80)
        print(output)
    
    # 统计信息
    if args.format == "text":
        print("\n" + "-"*80)
        print(f"Total results: {len(leaderboard)}")
        if leaderboard:
            avg_overall = sum(r["Overall"] for r in leaderboard) / len(leaderboard)
            best = max(leaderboard, key=lambda x: x["Overall"])
            print(f"Average Overall: {avg_overall:.2f}")
            print(f"Best: {best['Config ID']} ({best['Model']}) - {best['Overall']:.2f}")


if __name__ == "__main__":
    main()
