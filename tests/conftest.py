"""
tests/conftest.py

Lets the unit tests run on a lightweight install (CI) without torch /
sentence-transformers. If the real package is installed it is used untouched.
The stub raises if anything tries to build a real embedding model, so a test
that needs one fails loudly instead of silently using a fake.
"""

import sys
import types

try:
    import sentence_transformers  # noqa: F401
except ImportError:
    stub = types.ModuleType("sentence_transformers")

    class SentenceTransformer:  # noqa: D401
        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "sentence-transformers is not installed; this test needs the real "
                "embedding model (integration test)."
            )

    stub.SentenceTransformer = SentenceTransformer
    sys.modules["sentence_transformers"] = stub
