import asyncio
import sys
from pathlib import Path

# Support both package execution (python -m src) and direct debugging
try:
    from .main import main
except ImportError:
    # Running directly (debugging) - add parent to path for absolute imports
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.main import main

asyncio.run(main())
