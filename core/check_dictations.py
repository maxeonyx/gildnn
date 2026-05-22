"""Check for new or modified dictation files since last seen."""
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DICTATIONS_DIR = REPO_ROOT / "dictations"
SEEN_FILE = REPO_ROOT / "runs" / "dictations_seen.json"


def check() -> list[str]:
    """Return list of new/modified dictation filenames."""
    if not SEEN_FILE.exists():
        # First run — treat everything as new
        return sorted(f.name for f in DICTATIONS_DIR.glob("*.md"))

    seen = json.loads(SEEN_FILE.read_text())
    new_or_modified = []
    for f in sorted(DICTATIONS_DIR.glob("*.md")):
        mtime = os.path.getmtime(f)
        if f.name not in seen or mtime > seen[f.name]:
            new_or_modified.append(f.name)
    return new_or_modified


def mark_seen() -> None:
    """Update the seen state to current."""
    state = {}
    for f in sorted(DICTATIONS_DIR.glob("*.md")):
        state[f.name] = os.path.getmtime(f)
    SEEN_FILE.parent.mkdir(parents=True, exist_ok=True)
    SEEN_FILE.write_text(json.dumps(state, indent=2))


if __name__ == "__main__":
    new_files = check()
    if new_files:
        print(f"NEW DICTATIONS ({len(new_files)}):")
        for name in new_files:
            print(f"  {name}")
        if "--mark" in sys.argv:
            mark_seen()
            print("(marked as seen)")
    else:
        print("No new dictations.")
