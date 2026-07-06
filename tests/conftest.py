import sys
from pathlib import Path
import pytest

# Ensure backend root is in search path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from main import app

@pytest.fixture(autouse=True)
def configure_limiter(request):
    """Disable rate limiting by default for tests to prevent 429 failures.
    
    Enable specifically for tests marked with @pytest.mark.enable_rate_limiting.
    """
    if "enable_rate_limiting" in request.keywords:
        app.state.limiter.enabled = True
    else:
        app.state.limiter.enabled = False
    yield
    # Reset back to enabled by default
    app.state.limiter.enabled = True
