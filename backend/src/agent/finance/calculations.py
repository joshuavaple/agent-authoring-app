from typing import Union
import pandas as pd
import numpy as np


def _validate_rate(rate: Union[float, int]):
    if not isinstance(rate, (float, int)):
        raise ValueError("rate must be a number")
    if rate > 1 or rate < 0:
        raise ValueError("rate must be between 0 (0%) and 1 (100%)")


def _validate_money_amount(amount: Union[float, int]):
    if not isinstance(amount, (float, int)):
        raise ValueError("money amount must be a number")
    if amount < 0:
        raise ValueError("money amount must be positive")


def _pmt(rate: float, nper: int, pv: Union[float, int], fv=0, when=0):
    """Replicates Excel's PMT function, calculates the payment for a loan based on constant payments and a constant interest rate.

    Args:
        rate: Interest rate per period.
        nper: Total number of payment periods.
        pv: Present value (loan amount, positive if borrowed).
        fv: Future value (balance after last payment), default 0.
        when: 0 = payments at end of period (ordinary annuity, default),
            1 = payments at start of period (annuity due).

    Returns:
        The payment per period (negative = cash outflow, matching Excel's sign convention).
    """
    _validate_rate(rate)
    _validate_money_amount(pv)
    _validate_money_amount(fv)
    if rate == 0:
        return -(pv + fv) / nper

    factor = (1 + rate) ** nper
    return -(rate * (fv + pv * factor)) / ((1 + rate * when) * (factor - 1))
