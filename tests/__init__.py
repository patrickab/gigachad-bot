"""Test package. Importing it strips the host's environment, before conftest or any app module reads it.

Hermetic runs: what the shell around pytest exports (run.sh, the production unit, a Pi session, the factory engine)
must not reach config or the scripts the tests start, e.g. GIGACHAD_BASE_DIR pointing at real data.
POSTGRES_TEST_DATABASE_URL is the one input the suite takes.
"""

import os
import tempfile

_HOST_PREFIXES = ("GIGACHAD_", "PI_", "POSTGRES_PASSWORD_", "LEARNHARNESS_", "AGENT_SANDBOX_", "FACTORY_")
_HOST_VARIABLES = ("MINERU_SERVER_URL", "OLLAMA_BASE_URL", "OPENAI_API_KEY", "BRAVE_API_KEY", "JUPYTER_PATH", "VIRTUAL_ENV")

for _name in [name for name in os.environ if name.startswith(_HOST_PREFIXES) or name in _HOST_VARIABLES]:
    del os.environ[_name]

# Memory-store roots (the host's `~/.pi/memory_store`, the checkout's `.memory_store`) point into a throwaway folder.
_MEMORY_DIR = tempfile.mkdtemp(prefix="gigachad-test-memory-")
os.environ["MEMORY_STORE_ROOT"] = os.path.join(_MEMORY_DIR, "host")
os.environ["GIGACHAD_MEMORY_DIR"] = os.path.join(_MEMORY_DIR, "logins")
