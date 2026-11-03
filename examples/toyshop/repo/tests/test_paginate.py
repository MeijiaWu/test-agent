from shop.paginate import page_count, paginate


def test_second_page():
    assert paginate(list(range(30)), page=2, limit=10) == list(range(10, 20))


def test_page_count():
    assert page_count(25, limit=10) == 3
