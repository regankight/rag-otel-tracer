import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# Tests must not depend on the OpenTelemetry settings of whoever runs them. Cleared at
# import, before any test module calls configure(). Tests that need one set it themselves.
for _name in [n for n in os.environ if n.startswith("OTEL_")]:
    del os.environ[_name]
