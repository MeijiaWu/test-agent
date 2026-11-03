import pytest

from shop.cart import Cart


def test_add_accumulates():
    c = Cart()
    c.add("a")
    c.add("a", 2)
    assert c.items == {"a": 3}


def test_add_rejects_non_positive():
    with pytest.raises(ValueError):
        Cart().add("a", 0)
