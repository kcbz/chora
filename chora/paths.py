"""Keep installed package resources separate from the local data workspace."""
import os
from pathlib import Path

PACKAGE_ROOT=Path(__file__).resolve().parent
WORKSPACE_ROOT=Path(os.environ.get('CHORA_HOME',PACKAGE_ROOT.parent)).expanduser().resolve()
