"""Every Streamlit page must render without exceptions against the doubles.

Regression guard for a real crash: `ui/product_changes.py` called `path.name` on the
`str` that `PulseMindAgent.record_product_change` returns. These tests import each
page module with a stubbed `agent` block appended (the same trick the app's own
Streamlit runtime effectively uses: the module body runs top-to-bottom with `render`
available), run them through `streamlit.testing.v1.AppTest.from_file`, and fail if
any page raises. No LLM or Hindsight server is needed: pages receive the fakes from
tests/test_agent.py, and UI actions that would call them are not triggered.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from core.config import get_settings
from core.hindsight_client import RetainReceipt
from services import trend_analyzer as trends

st = pytest.importorskip("streamlit.testing.v1")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
UI_DIR = PROJECT_ROOT / "ui"
STUB = '''

# ---- test stub (appended by tests/test_ui_pages.py) ----
import sys as _sys

_sys.path.insert(0, r"{project_root}")

from core.agent import PulseMindAgent as PulseMindAgent  # noqa: E402
from core.config import get_settings as get_settings  # noqa: E402
from tests.test_agent import FakeLLM as FakeLLM  # noqa: E402
from tests.test_agent import FakeMemory as FakeMemory  # noqa: E402

_settings = get_settings()
agent = PulseMindAgent(_settings, memory=FakeMemory(_settings), llm=FakeLLM())
agent.reload_data()
render(agent)
'''


@pytest.fixture(scope="module")
def app_test_factory():
    """Import the real doubles once, then build runnable copies of each page."""

    from tests.test_agent import FakeLLM, FakeMemory  # noqa: PLC0415

    def factory(page_name: str):
        source = (UI_DIR / f"{page_name}.py").read_text(encoding="utf-8")
        runnable = Path(f".render_{page_name}.py")
        stub = STUB.format(project_root=PROJECT_ROOT)
        runnable.write_text(source + stub, encoding="utf-8")
        return runnable

    return factory


@pytest.mark.parametrize(
    "page_name",
    [
        "dashboard",
        "feedback_explorer",
        "product_changes",
        "insights",
        "memory_explorer",
        "ask_pulsemind",
    ],
)
def test_page_renders_without_exception(app_test_factory, page_name: str) -> None:
    runnable = app_test_factory(page_name)
    try:
        # AppTest.from_file resolves relative paths against the caller's directory,
        # so pass the absolute path of the generated script. The timeout is explicit
        # because the first AppTest run in a process also pays Streamlit's one-off
        # component-manifest scan (~4s against a full site-packages), which is far
        # longer than AppTest's 3s default and has nothing to do with the page.
        result = st.AppTest.from_file(str(runnable.resolve())).run(timeout=60)
        assert not result.exception, f"{page_name} raised: {result.exception}"
    finally:
        runnable.unlink(missing_ok=True)
