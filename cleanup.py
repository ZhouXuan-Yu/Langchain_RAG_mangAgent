"""Clean up temporary test files."""
import os
files_to_remove = [
    "verify_fix.py",
    "check_sync.py",
    "run_verify.bat",
    "cleanup.bat",
]
removed = []
for f in files_to_remove:
    if os.path.exists(f):
        os.remove(f)
        removed.append(f)
print(f"Cleaned: {removed}")
