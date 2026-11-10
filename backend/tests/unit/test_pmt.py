import pytest
from agent.finance.calculations import _pmt


def test_pmt_matches_excel_example():
    # Excel: =PMT(5%/12, 300, 560000) -> -3273.70
    result = _pmt(rate=0.05 / 12, nper=300, pv=560000)
    assert result == pytest.approx(-3273.7042324446884)
