"""
conftest.py — pytest configuration for the mlops-task test suite.
Ensures the project root is on sys.path so tests can import run.py directly.
"""
import sys
from pathlib import Path

# Add project root to path (one level above the tests/ directory)
sys.path.insert(0, str(Path(__file__).parent.parent))
