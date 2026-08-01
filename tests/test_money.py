import pytest
from hypothesis import given
from hypothesis import strategies as st

from datagraph.money import AllocationError, allocate

# The invariant the whole settlement path rests on: credits are conserved exactly.
finite_weights = st.lists(
    st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False),
    min_size=1,
    max_size=24,
)


@given(total=st.integers(min_value=0, max_value=10**9), weights=finite_weights)
def test_allocation_sums_to_total_exactly(total, weights):
    assert sum(allocate(total, weights)) == total


# The range above is comfortable; these cover the whole range the function actually accepts.
# Floating-point apportionment passed the comfortable range and failed here, which is the
# argument for testing to the contract rather than to the happy path.
extreme_weights = st.lists(
    st.floats(min_value=0.0, allow_nan=False, allow_infinity=False),
    min_size=1,
    max_size=8,
)


@given(total=st.integers(min_value=0, max_value=10**30), weights=extreme_weights)
def test_allocation_is_exact_across_the_full_accepted_range(total, weights):
    assert sum(allocate(total, weights)) == total


@pytest.mark.parametrize(
    ("total", "weights"),
    [
        (100, [1e308, 1e308]),  # the sum of these overflows a float to inf
        (10**18, [1, 2, 3]),  # beyond 2^53, so float shares lose integer precision
        (10**30, [1e-300, 1.0]),  # ratio too extreme for float shares to resolve
    ],
)
def test_allocation_survives_values_that_break_float_arithmetic(total, weights):
    result = allocate(total, weights)
    assert sum(result) == total
    assert all(share >= 0 for share in result)


@given(total=st.integers(min_value=0, max_value=10**6), weights=finite_weights)
def test_allocation_is_never_negative(total, weights):
    assert all(share >= 0 for share in allocate(total, weights))


@given(total=st.integers(min_value=0, max_value=10**6), weights=finite_weights)
def test_allocation_is_deterministic(total, weights):
    assert allocate(total, weights) == allocate(total, weights)


def test_proportional_split_is_exact_when_it_divides_evenly():
    assert allocate(100, [1, 1, 1, 1]) == [25, 25, 25, 25]
    assert allocate(90, [2, 1]) == [60, 30]


def test_largest_remainder_distributes_the_leftover():
    # 100 / 3 leaves one credit over; it goes to the largest remainder, ties broken by index.
    result = allocate(100, [1, 1, 1])
    assert sum(result) == 100
    assert sorted(result) == [33, 33, 34]
    assert result == [34, 33, 33]


def test_zero_weight_earns_nothing():
    # The property that makes attribution meaningful: contribute nothing, earn nothing.
    assert allocate(100, [1, 0, 1]) == [50, 0, 50]


def test_all_zero_weights_split_equally():
    # Documented limit behaviour. Callers that care about the *policy* must check first —
    # see test_marketplace.py for the refund path this exists to make visible.
    assert allocate(99, [0, 0, 0]) == [33, 33, 33]


def test_tiny_total_across_many_recipients():
    result = allocate(1, [1, 1, 1, 1])
    assert sum(result) == 1
    assert result.count(1) == 1


def test_zero_total_pays_nobody():
    assert allocate(0, [5, 3, 2]) == [0, 0, 0]


@pytest.mark.parametrize(
    ("total", "weights"),
    [
        (-1, [1.0]),
        (10, []),
        (10, [1.0, -0.5]),
        (10, [float("nan")]),
        (10, [float("inf")]),
    ],
)
def test_invalid_input_is_rejected(total, weights):
    with pytest.raises(AllocationError):
        allocate(total, weights)
