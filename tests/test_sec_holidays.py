"""EDGAR holidays: no index request, an empty index (no network)."""
import pytest

from core import sec


def test_federal_holidays_on_market_days():
    assert sec.edgar_holiday("2026-10-12")           # Columbus Day: market open, EDGAR closed
    assert sec.edgar_holiday("2026-11-11")           # Veterans Day
    assert not sec.edgar_holiday("2026-10-13")


def test_holiday_index_is_empty_without_a_request(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("no request on a holiday")
    monkeypatch.setattr(sec, "get", boom)
    idx = sec.daily_form_index("2026-10-12")
    assert len(idx) == 0 and list(idx.columns) == ["form", "company", "cik", "filed", "file"]
    with pytest.raises(AssertionError):
        sec.daily_form_index("2026-10-13")
