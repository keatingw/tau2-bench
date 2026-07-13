import hashlib
import importlib.metadata
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

from deepdiff import DeepDiff
from dotenv import load_dotenv
from loguru import logger

res = load_dotenv()
if not res:
    logger.warning("No .env file found")

# Try to get data directory from environment variable first
DATA_DIR_ENV = os.getenv("TAU2_DATA_DIR")

if DATA_DIR_ENV:
    # Use environment variable if set
    DATA_DIR = Path(DATA_DIR_ENV)
    logger.info(f"Using data directory from environment: {DATA_DIR}")
else:
    # Fallback to source directory (for development)
    SOURCE_DIR = Path(__file__).parents[3]
    DATA_DIR = SOURCE_DIR / "data"
    logger.info(f"Using data directory from source: {DATA_DIR}")

# Check if data directory exists and is accessible
if not DATA_DIR.exists():
    logger.warning(f"Data directory does not exist: {DATA_DIR}")
    logger.warning(
        "Set TAU2_DATA_DIR environment variable to point to your data directory"
    )
    logger.warning("Or ensure the data directory exists in the expected location")


def canonicalize_json_numbers(obj):
    """
    Recursively convert integral floats to ints (33.0 -> 33) so that values
    which are numerically equal serialize to the same JSON text.

    JSON does not distinguish int from float, but json.dumps renders 33 and
    33.0 differently, so two semantically identical structures can hash
    differently depending only on how a number was formatted upstream (e.g.
    an LLM emitting `33` vs `33.00` in a tool call). Bools are untouched
    (bool is an int subclass but a distinct JSON type).
    """
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, float) and obj.is_integer():
        return int(obj)
    if isinstance(obj, dict):
        return {k: canonicalize_json_numbers(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [canonicalize_json_numbers(v) for v in obj]
    return obj


def get_dict_hash(obj: dict) -> str:
    """
    Generate a unique hash for dict.
    Returns a hex string representation of the hash.

    Numbers are canonicalized before hashing so that structures differing
    only in int-vs-integral-float formatting (33 vs 33.0) hash identically.
    """
    hash_string = json.dumps(canonicalize_json_numbers(obj), sort_keys=True, default=str)
    return hashlib.sha256(hash_string.encode()).hexdigest()


def show_dict_diff(dict1: dict, dict2: dict) -> str:
    """
    Show the difference between two dictionaries.
    """
    diff = DeepDiff(dict1, dict2)
    return diff


def get_now(use_compact_format: bool = False) -> str:
    """
    Returns the current date and time.

    Args:
        use_compact_format: If True, returns format YYYYMMDD_HHMMSS.
                          If False, returns ISO format (YYYY-MM-DDTHH:MM:SS.ffffff).
    """
    now = datetime.now()
    return format_time(now, use_compact_format=use_compact_format)


def format_time(time: datetime, use_compact_format: bool = True) -> str:
    """
    Format the time.

    Args:
        time: The datetime object to format.
        use_compact_format: If True, returns format YYYYMMDD_HHMMSS.
                          If False, returns ISO format (YYYY-MM-DDTHH:MM:SS.ffffff).
    """
    if use_compact_format:
        return time.strftime("%Y%m%d_%H%M%S")
    else:
        return time.isoformat()


def get_tau2_version() -> str:
    """Get the installed tau2 package version."""
    try:
        return importlib.metadata.version("tau2")
    except importlib.metadata.PackageNotFoundError:
        return "dev"


def get_commit_hash() -> str:
    """
    Get the commit hash of the current directory.
    """
    try:
        commit_hash = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], text=True)
            .strip()
            .split("\n")[0]
        )
    except Exception as e:
        logger.error(f"Failed to get git hash: {e}")
        commit_hash = "unknown"
    return commit_hash
