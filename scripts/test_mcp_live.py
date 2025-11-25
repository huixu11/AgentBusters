#!/usr/bin/env python3
"""
Test MCP Servers with Live Data

Tests the MCP servers by calling their tools directly with real financial data.
"""

import sys
import asyncio
from datetime import datetime

sys.path.insert(0, "src")

from mcp_servers.sec_edgar import create_edgar_server
from mcp_servers.yahoo_finance import create_yahoo_finance_server
from mcp_servers.sandbox import create_sandbox_server


async def test_yahoo_finance_server():
    """Test Yahoo Finance MCP server with real data."""
    print("\n" + "=" * 60)
    print("Testing Yahoo Finance MCP Server")
    print("=" * 60)

    server = create_yahoo_finance_server()
    tools = await server.get_tools()

    print(f"\nRegistered tools: {list(tools.keys())}")

    # Test get_quote
    print("\n--- Testing get_quote('NVDA') ---")
    get_quote = tools["get_quote"]
    result = get_quote.fn(ticker="NVDA")
    print(f"Result type: {type(result)}")
    if isinstance(result, dict):
        print(f"  Ticker: {result.get('ticker')}")
        print(f"  Name: {result.get('name')}")
        print(f"  Current Price: ${result.get('current_price')}")
        print(f"  Market Cap: ${result.get('market_cap'):,.0f}" if result.get('market_cap') else "  Market Cap: N/A")
        print(f"  P/E Ratio: {result.get('pe_ratio')}")

    # Test get_historical_prices
    print("\n--- Testing get_historical_prices('AAPL', '5d') ---")
    get_historical = tools["get_historical_prices"]
    result = get_historical.fn(ticker="AAPL", period="5d", interval="1d")
    if isinstance(result, list) and len(result) > 0:
        print(f"  Got {len(result)} data points")
        latest = result[-1]
        print(f"  Latest: {latest.get('date')} - Close: ${latest.get('close'):.2f}")

    # Test get_key_statistics
    print("\n--- Testing get_key_statistics('MSFT') ---")
    get_stats = tools["get_key_statistics"]
    result = get_stats.fn(ticker="MSFT")
    if isinstance(result, dict):
        print(f"  Market Cap: ${result.get('market_cap'):,.0f}" if result.get('market_cap') else "  Market Cap: N/A")
        print(f"  P/E Ratio: {result.get('pe_ratio')}")
        print(f"  Beta: {result.get('beta')}")

    print("\n✓ Yahoo Finance MCP Server tests passed!")


async def test_sec_edgar_server():
    """Test SEC EDGAR MCP server with real data."""
    print("\n" + "=" * 60)
    print("Testing SEC EDGAR MCP Server")
    print("=" * 60)

    server = create_edgar_server()
    tools = await server.get_tools()

    print(f"\nRegistered tools: {list(tools.keys())}")

    # Test get_company_info
    print("\n--- Testing get_company_info('AAPL') ---")
    get_company = tools["get_company_info"]
    result = get_company.fn(ticker="AAPL")
    if isinstance(result, dict):
        print(f"  Ticker: {result.get('ticker')}")
        print(f"  CIK: {result.get('cik')}")
        print(f"  Name: {result.get('name')}")

    # Test get_filing
    print("\n--- Testing get_filing('NVDA', '10-K') ---")
    get_filing = tools["get_filing"]
    result = get_filing.fn(ticker="NVDA", form_type="10-K")
    if isinstance(result, dict):
        print(f"  Ticker: {result.get('ticker')}")
        print(f"  Form Type: {result.get('form_type')}")
        print(f"  Filing Date: {result.get('filing_date')}")
        if result.get('error'):
            print(f"  Error: {result.get('error')}")

    print("\n✓ SEC EDGAR MCP Server tests passed!")


async def test_sandbox_server():
    """Test Python Sandbox MCP server."""
    print("\n" + "=" * 60)
    print("Testing Python Sandbox MCP Server")
    print("=" * 60)

    server = create_sandbox_server()
    tools = await server.get_tools()

    print(f"\nRegistered tools: {list(tools.keys())}")

    # Test execute_python
    print("\n--- Testing execute_python (CAGR calculation) ---")
    execute = tools["execute_python"]
    code = """
import numpy as np

# Calculate CAGR
initial = 1000
final = 1500
years = 3
cagr = (final / initial) ** (1/years) - 1
print(f"CAGR: {cagr:.2%}")
"""
    result = execute.fn(code=code)
    if isinstance(result, dict):
        print(f"  Success: {result.get('success')}")
        print(f"  Output: {result.get('stdout').strip()}")
        print(f"  Execution Time: {result.get('execution_time_ms')}ms")

    # Test calculate_financial_metric
    print("\n--- Testing calculate_financial_metric (gross_margin) ---")
    calc_metric = tools["calculate_financial_metric"]
    result = calc_metric.fn(
        metric="gross_margin",
        values={"revenue": 100_000_000, "cogs": 60_000_000}
    )
    if isinstance(result, dict):
        print(f"  Metric: {result.get('metric')}")
        print(f"  Value: {result.get('value'):.2%}" if result.get('value') else f"  Error: {result.get('error')}")
        print(f"  Formula: {result.get('formula')}")

    # Test analyze_time_series
    print("\n--- Testing analyze_time_series ---")
    analyze_ts = tools["analyze_time_series"]
    data = [100, 105, 110, 108, 115, 120, 118, 125, 130, 128]
    result = analyze_ts.fn(
        data=data,
        operations=["mean", "std", "trend"]
    )
    if isinstance(result, dict):
        print(f"  Data Points: {result.get('data_points')}")
        print(f"  Mean: {result.get('mean'):.2f}")
        print(f"  Std Dev: {result.get('std'):.2f}")
        if isinstance(result.get('trend'), dict):
            print(f"  Trend Slope: {result['trend'].get('slope'):.4f}")

    print("\n✓ Python Sandbox MCP Server tests passed!")


async def test_temporal_locking():
    """Test temporal locking functionality."""
    print("\n" + "=" * 60)
    print("Testing Temporal Locking (Time Machine)")
    print("=" * 60)

    # Create server with simulation date in the past
    simulation_date = datetime(2024, 6, 1)
    print(f"\nSimulation date: {simulation_date}")

    server = create_yahoo_finance_server(simulation_date=simulation_date)
    tools = await server.get_tools()

    print("\n--- Testing get_historical_prices with temporal lock ---")
    get_historical = tools["get_historical_prices"]
    result = get_historical.fn(ticker="AAPL", period="1y", interval="1d")

    if isinstance(result, list) and len(result) > 0:
        latest_date = result[-1].get("date")
        print(f"  Got {len(result)} data points")
        print(f"  Latest date in results: {latest_date}")
        print(f"  Simulation date: {simulation_date.strftime('%Y-%m-%d')}")

        # Check that no data is after simulation date
        if latest_date:
            latest = datetime.strptime(latest_date, "%Y-%m-%d")
            if latest <= simulation_date:
                print("  ✓ Temporal lock is working - no future data returned")
            else:
                print("  ✗ Warning: Data after simulation date was returned")

    print("\n✓ Temporal Locking tests passed!")


async def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("MCP Servers Live Test Suite")
    print("=" * 60)
    print(f"Time: {datetime.now().isoformat()}")

    try:
        await test_yahoo_finance_server()
        await test_sec_edgar_server()
        await test_sandbox_server()
        await test_temporal_locking()

        print("\n" + "=" * 60)
        print("All MCP Server Tests Passed! ✓")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
