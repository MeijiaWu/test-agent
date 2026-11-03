paginate() ignores the documented default page size

The docstring of `shop.paginate.paginate` says `limit` defaults to `DEFAULT_LIMIT` (10),
but calling `paginate(items)` without `limit` returns 50 items:

    >>> from shop.paginate import paginate
    >>> len(paginate(list(range(100))))
    50

Expected 10.
