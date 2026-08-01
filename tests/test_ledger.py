import pytest
from hypothesis import given
from hypothesis import strategies as st

from datagraph.ledger import EXTERNAL, Ledger, LedgerError
from datagraph.money import allocate


def test_funding_creates_credits_only_at_the_boundary():
    ledger = Ledger()
    ledger.fund("researcher:r1", 1000)

    assert ledger.balance("researcher:r1") == 1000
    assert ledger.balance(EXTERNAL) == -1000
    assert ledger.credits_in_circulation() == 1000
    ledger.check_invariants()


def test_unbalanced_entry_is_rejected():
    ledger = Ledger()
    with pytest.raises(LedgerError, match="does not balance"):
        ledger.post("bad", {"a": 10, "b": -5})


def test_account_cannot_be_overdrawn():
    ledger = Ledger()
    ledger.fund("researcher:r1", 100)
    with pytest.raises(LedgerError, match="would overdraw"):
        ledger.open_escrow("escrow:q1", "researcher:r1", 101)


def test_full_escrow_lifecycle_conserves_credits():
    ledger = Ledger()
    ledger.fund("researcher:r1", 500)
    ledger.open_escrow("escrow:q1", "researcher:r1", 300)

    assert ledger.balance("researcher:r1") == 200
    assert ledger.balance("escrow:q1") == 300
    ledger.check_invariants()

    ledger.settle_escrow("escrow:q1", {"provider:p1": 200, "provider:p2": 100})

    assert ledger.balance("escrow:q1") == 0
    assert ledger.balance("provider:p1") == 200
    assert ledger.balance("provider:p2") == 100
    assert ledger.credits_in_circulation() == 500
    assert ledger.open_escrows() == {}
    ledger.check_invariants()


def test_refund_returns_the_whole_escrow():
    ledger = Ledger()
    ledger.fund("researcher:r1", 500)
    ledger.open_escrow("escrow:q1", "researcher:r1", 300)
    ledger.refund_escrow("escrow:q1", "researcher:r1")

    assert ledger.balance("researcher:r1") == 500
    assert ledger.balance("escrow:q1") == 0
    assert ledger.open_escrows() == {}
    ledger.check_invariants()


def test_settlement_that_does_not_exhaust_the_escrow_is_rejected():
    # The invariant that stops an attribution bug from silently losing money.
    ledger = Ledger()
    ledger.fund("researcher:r1", 500)
    ledger.open_escrow("escrow:q1", "researcher:r1", 300)

    with pytest.raises(LedgerError, match="sum to 299 but the escrow holds 300"):
        ledger.settle_escrow("escrow:q1", {"provider:p1": 299})

    # The failed settlement changed nothing.
    assert ledger.balance("escrow:q1") == 300
    ledger.check_invariants()


def test_settlement_that_overspends_the_escrow_is_rejected():
    ledger = Ledger()
    ledger.fund("researcher:r1", 500)
    ledger.open_escrow("escrow:q1", "researcher:r1", 300)

    with pytest.raises(LedgerError, match="sum to 301 but the escrow holds 300"):
        ledger.settle_escrow("escrow:q1", {"provider:p1": 301})
    ledger.check_invariants()


def test_negative_payout_is_rejected():
    ledger = Ledger()
    ledger.fund("researcher:r1", 500)
    ledger.open_escrow("escrow:q1", "researcher:r1", 300)

    with pytest.raises(LedgerError, match="is negative"):
        ledger.settle_escrow("escrow:q1", {"provider:p1": 400, "provider:p2": -100})
    ledger.check_invariants()


def test_escrow_cannot_be_opened_twice_or_settled_when_closed():
    ledger = Ledger()
    ledger.fund("researcher:r1", 500)
    ledger.open_escrow("escrow:q1", "researcher:r1", 100)

    with pytest.raises(LedgerError, match="already open"):
        ledger.open_escrow("escrow:q1", "researcher:r1", 100)

    ledger.refund_escrow("escrow:q1", "researcher:r1")
    with pytest.raises(LedgerError, match="is not open"):
        ledger.settle_escrow("escrow:q1", {"provider:p1": 100})


@given(
    funding=st.integers(min_value=1, max_value=10**6),
    payments=st.lists(st.integers(min_value=1, max_value=1000), min_size=1, max_size=20),
    weights=st.lists(
        st.floats(min_value=0.0, max_value=1e3, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=6,
    ),
)
def test_credits_are_conserved_across_arbitrary_query_sequences(funding, payments, weights):
    """The end-to-end invariant: no sequence of escrow-and-settle creates or destroys credits."""
    ledger = Ledger()
    ledger.fund("researcher:r1", funding)

    providers = [f"provider:p{i}" for i in range(len(weights))]

    for i, payment in enumerate(payments):
        if ledger.balance("researcher:r1") < payment:
            continue

        escrow = f"escrow:q{i}"
        ledger.open_escrow(escrow, "researcher:r1", payment)

        if sum(weights) == 0:
            ledger.refund_escrow(escrow, "researcher:r1")
        else:
            shares = allocate(payment, weights)
            ledger.settle_escrow(escrow, dict(zip(providers, shares, strict=True)))

        ledger.check_invariants()

    assert ledger.credits_in_circulation() == funding
    assert ledger.open_escrows() == {}
