import os
from pathlib import Path

# Resolve the backend root directory (parent of app/)
_BACKEND_ROOT = Path(__file__).resolve().parent.parent.parent


def get_data_dir() -> Path:
    """Returns the writable data directory for all persistent storage.

    On Render.com (and similar PaaS hosts), the DATA_DIR environment variable
    should point to the persistent disk mount path (e.g. /data). Locally,
    this falls back to a data/ subdirectory inside the backend folder.

    Returns:
        An absolute Path to the data root directory.
    """
    env_data_dir = os.getenv("DATA_DIR")
    if env_data_dir:
        return Path(env_data_dir).resolve()
    return _BACKEND_ROOT / "data"
