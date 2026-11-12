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


def _amortization_schedule(
    principal: Union[float, int],
    annual_rate: Union[float, int],
    nper: int,
    periods_per_year: int = 12,
    fv: Union[float, int] = 0,
    when=0,
) -> pd.DataFrame:
    """Build a period-by-period amortization schedule.

    Args:
        principal: Loan amount (positive).
        annual_rate: Nominal annual interest rate (e.g. 0.06 for 6%).
        nper: Total number of payment periods.
        periods_per_year: Compounding/payment frequency (12 = monthly).
        fv: Target balance at the end (0 for fully amortizing).
        when: 0 = end-of-period payments, 1 = start-of-period.

    Returns:
        A DataFrame indexed by period, with columns: payment, principal_paid,
        interest_paid, ending_balance.
    """
    _validate_rate(annual_rate)
    _validate_money_amount(principal)

    rate = annual_rate / periods_per_year
    payment = _pmt(rate, nper, principal, fv, when)  # negative, Excel convention

    periods = np.arange(1, nper + 1)
    balance = principal
    rows = []

    for p in periods:
        interest = balance * rate
        principal_paid = -payment - interest
        balance = balance + interest + payment  # payment is negative -> reduces balance
        rows.append((p, payment, principal_paid, interest, balance))

    df = pd.DataFrame(
        rows,
        columns=[
            "period",
            "payment",
            "principal_paid",
            "interest_paid",
            "ending_balance",
        ],
    )
    return df


def _property_tax_amount(
    rate: Union[float, int], assessed_value: Union[float, int]
) -> float:
    """Calculate the property tax amount of a home based on its assessed value.

    Args:
        rate: Property tax rate.
        assessed_value: Assessed value of the home.

    Returns:
        The property tax amount.
    """
    _validate_rate(rate)
    _validate_money_amount(assessed_value)
    return rate * assessed_value


def _gross_debt_service(
        mortgage_pmt, property_tax, utilities, other_expenses, gross_income
) -> float:
    """
    Calculate the gross debt service ratio (GDS) of a household in a period (e.g., monthly).
    Measures the percentage of your pre-tax income needed to cover your core housing costs.
    This is a affordability stress test to see if one's earning is enough to handle a specific home purchase without risking default.
    Args:
        mortgage_pmt: total mortgage payment in the period (principal and interest)
        property_tax: proprety tax amount in the period
        utilities: utilities amount in the period
        other_expenses: other household expenses like phone, internet and cable in the period
        gross_income: total income before tax of the individual(s) applying for the mortgage in the period
    
    Returns:
        The GDS ratio
    """
    for amount in [mortgage_pmt, property_tax, utilities, other_expenses, gross_income]:
        _validate_money_amount(amount)
    
    return sum([mortgage_pmt, property_tax, utilities, other_expenses])/gross_income