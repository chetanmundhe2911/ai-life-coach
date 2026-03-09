import hashlib
from pathlib import Path
from typing import List
from datetime import datetime

def get_file_hash(filepath: str) -> str:
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

def get_supported_files(directory: Path) -> List[Path]:
    supported_extensions = {".txt", ".md", ".pdf"}
    files = []
    if not directory.exists():
        return files
    for file_path in directory.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
            files.append(file_path)
    return sorted(files)

def truncate_text(text: str, max_chars: int = 200) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."

def count_tokens_estimate(text: str) -> int:
    return len(text) // 4