"""
A simple MCP server that provides math tools.
"""
import sys
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("MathTools")

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers."""
    print(f"Frank: Adding {a} and {b}", file=sys.stderr)
    return a + b

@mcp.tool()
def subtract(a: int, b: int) -> int:
    """Subtract b from a."""
    print(f"Frank: Subtracting {b} from {a}", file=sys.stderr)
    return a - b

@mcp.tool()
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    print(f"Frank: Multiplying {a} and {b}", file=sys.stderr)
    return a * b

@mcp.tool()
def divide(a: float, b: float) -> float:
    """Divide a by b. Returns an error if b is 0."""
    print(f"Frank: Dividing {a} by {b}", file=sys.stderr)
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b

if __name__ == "__main__":
    mcp.run(transport="stdio")
