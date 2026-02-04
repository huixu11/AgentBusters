#!/usr/bin/env python
"""
View detailed interactions from an evaluation result file.

Usage:
    python scripts/view_interactions.py results/eval_xxx.json
    python scripts/view_interactions.py results/eval_xxx.json --step 0  # Show specific step
    python scripts/view_interactions.py results/eval_xxx.json --trades  # Show trades only

To record interactions, run eval with RECORD_INTERACTIONS=1:
    $env:RECORD_INTERACTIONS="1"; python scripts/run_a2a_eval.py ...
"""

import argparse
import json
import sys
from pathlib import Path


def format_payload(payload: dict) -> str:
    """Format a trading decision payload in a readable way."""
    state = payload.get("state", {})
    ohlcv = state.get("ohlcv", {})
    account = state.get("account", {})
    indicators = state.get("indicators", {})
    
    lines = [
        f"  Symbol: {state.get('symbol', 'N/A')}",
        f"  Timestamp: {state.get('timestamp', 'N/A')}",
        f"  OHLCV: O={ohlcv.get('open'):.2f} H={ohlcv.get('high'):.2f} L={ohlcv.get('low'):.2f} C={ohlcv.get('close'):.2f}",
        f"  Volume: {ohlcv.get('volume', 0):,.0f}",
        f"  Account: Balance=${account.get('balance', 0):,.2f}, Equity=${account.get('equity', 0):,.2f}",
    ]
    
    if account.get("positions"):
        for pos in account["positions"]:
            lines.append(f"  Position: {pos.get('size', 0):.4f} @ ${pos.get('entry_price', 0):.2f}")
            if pos.get("stop_loss"):
                lines.append(f"    Stop Loss: ${pos['stop_loss']:.2f}")
            if pos.get("take_profit"):
                lines.append(f"    Take Profit: ${pos['take_profit']:.2f}")
    
    if indicators:
        ind_str = ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" 
                          for k, v in list(indicators.items())[:5])
        lines.append(f"  Indicators: {ind_str}")
    
    return "\n".join(lines)


def format_decision(decision: dict) -> str:
    """Format a parsed decision in a readable way."""
    lines = [
        f"  Action: {decision.get('action', 'N/A')}",
        f"  Size: {decision.get('size', 0):.4f}",
        f"  Confidence: {decision.get('confidence', 0):.2%}",
    ]
    
    if decision.get("stop_loss"):
        lines.append(f"  Stop Loss: ${decision['stop_loss']:.2f}")
    if decision.get("take_profit"):
        lines.append(f"  Take Profit: ${decision['take_profit']:.2f}")
    if decision.get("reasoning"):
        reasoning = decision["reasoning"][:200] + "..." if len(decision.get("reasoning", "")) > 200 else decision.get("reasoning", "")
        lines.append(f"  Reasoning: {reasoning}")
    
    return "\n".join(lines)


def view_interactions(result_file: Path, step: int | None = None, show_trades: bool = False):
    """View interactions from a result file."""
    with open(result_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Navigate to results
    results = None
    if "result" in data:
        # A2A format
        artifacts = data["result"].get("artifacts", [])
        for artifact in artifacts:
            for part in artifact.get("parts", []):
                if part.get("kind") == "data" and "results" in part.get("data", {}):
                    results = part["data"]["results"]
                    break
    elif "results" in data:
        results = data["results"]
    
    if not results:
        print("No results found in file")
        return
    
    for result in results:
        scenario_name = result.get("scenario_name", result.get("example_id", "unknown"))
        print(f"\n{'='*80}")
        print(f"Scenario: {scenario_name}")
        print(f"Score: {result.get('score', 0):.2f} (Grade: {result.get('grade', 'N/A')})")
        print(f"{'='*80}")
        
        # Show trades
        if show_trades or "trades" in result:
            trades = result.get("trades", [])
            if trades:
                print(f"\n📈 TRADES ({len(trades)} total):")
                print("-" * 60)
                for i, trade in enumerate(trades):
                    pnl_sign = "+" if trade["pnl"] >= 0 else ""
                    print(f"  Trade {i+1}: Entry ${trade['entry_price']:.2f} → Exit ${trade['exit_price']:.2f}")
                    print(f"           Size: {trade['size']:.4f}, PnL: {pnl_sign}${trade['pnl']:.2f} ({trade['reason']})")
            else:
                print("\n📈 No trades executed")
            
            if show_trades:
                continue
        
        # Show interactions
        interactions = result.get("interactions", [])
        if not interactions:
            print("\n⚠️  No interactions recorded. Run with RECORD_INTERACTIONS=1 to enable.")
            continue
        
        print(f"\n📝 INTERACTIONS ({len(interactions)} decision points):")
        
        for interaction in interactions:
            if step is not None and interaction.get("step") != step:
                continue
            
            print("\n" + "-" * 60)
            print(f"Step {interaction.get('step', '?')} | {interaction.get('timestamp', 'N/A')}")
            print("-" * 60)
            
            # Sent payload
            print("\n🔷 GREEN AGENT SENT:")
            if "sent_payload" in interaction:
                print(format_payload(interaction["sent_payload"]))
            
            # Error or response
            if "error" in interaction:
                print(f"\n❌ ERROR: {interaction['error']}")
            else:
                print("\n🔶 PURPLE AGENT RAW RESPONSE:")
                raw = interaction.get("raw_response", "")
                if len(raw) > 500:
                    print(f"  {raw[:500]}...")
                    print(f"  [... {len(raw) - 500} more chars ...]")
                else:
                    print(f"  {raw}")
            
            # Parsed decision
            print("\n🎯 PARSED DECISION:")
            if "parsed_decision" in interaction:
                print(format_decision(interaction["parsed_decision"]))
            
            if "equity_before" in interaction:
                print(f"\n💰 Equity at decision: ${interaction['equity_before']:,.2f}")


def main():
    parser = argparse.ArgumentParser(description="View detailed interactions from evaluation results")
    parser.add_argument("result_file", type=Path, help="Path to the result JSON file")
    parser.add_argument("--step", type=int, default=None, help="Show only a specific step")
    parser.add_argument("--trades", action="store_true", help="Show trades only")
    
    args = parser.parse_args()
    
    if not args.result_file.exists():
        print(f"Error: File not found: {args.result_file}")
        sys.exit(1)
    
    view_interactions(args.result_file, step=args.step, show_trades=args.trades)


if __name__ == "__main__":
    main()
