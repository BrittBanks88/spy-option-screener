"""Repo-root entry point so `streamlit run streamlit_app.py` and Streamlit
Community Cloud (which defaults to this filename) both just work.

The real app lives in spy_option_screener/app/streamlit_app.py; this re-execs
it on every Streamlit rerun.
"""
import runpy
from pathlib import Path

_REAL = Path(__file__).parent / "spy_option_screener" / "app" / "streamlit_app.py"
runpy.run_path(str(_REAL), run_name="__main__")
