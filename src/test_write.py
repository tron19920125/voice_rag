"""Simple example module with a multiply helper and inline tests."""


def multiply(a, b):
    """Return the product of a and b."""
    return a * b


def test_multiply_positive_numbers():
    assert multiply(2, 3) == 6


def test_multiply_with_zero():
    assert multiply(5, 0) == 0

