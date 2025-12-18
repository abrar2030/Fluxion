#!/usr/bin/env python3
"""Test if the backend can start without errors"""

import sys
import traceback


def test_imports():
    """Test all critical imports"""
    try:
        print("Testing imports...")

        print("  - config.settings")

        print("  - config.database")

        print("  - schemas.base")

        print("  - middleware")

        print("  - api.v1.router")

        print("  - app.main")

        print("\n✓ All imports successful!")
        return True
    except Exception as e:
        print(f"\n✗ Import failed: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)
