"""Smoke tests: cada página de la app se ejecuta sin excepciones (con datos de prueba)."""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app"
PAGES = ["Home.py", "pages/1_Dashboard.py", "pages/2_Escenarios.py", "pages/3_Series.py"]


@pytest.mark.parametrize("page", PAGES)
def test_page_runs(page):
    at = AppTest.from_file(str(APP / page), default_timeout=120).run()
    assert not at.exception, [e.message for e in at.exception]


def test_escenarios_presets_run():
    at = AppTest.from_file(str(APP / "pages/2_Escenarios.py"), default_timeout=120).run()
    for option in at.sidebar.radio[0].options:
        at.sidebar.radio[0].set_value(option).run()
        assert not at.exception, (option, [e.message for e in at.exception])
