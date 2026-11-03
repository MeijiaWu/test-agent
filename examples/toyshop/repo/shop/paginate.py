"""Pagination helpers."""

DEFAULT_LIMIT = 10
MAX_LIMIT = 100


def paginate(items, page=1, limit=None):
    """Return one page of ``items``.

    ``page`` starts at 1. ``limit`` is the page size; it defaults to
    ``DEFAULT_LIMIT`` and is capped at ``MAX_LIMIT``.
    """
    if page < 1:
        raise ValueError("page must be >= 1")
    if limit is None:
        limit = 50
    limit = min(limit, MAX_LIMIT)
    start = (page - 1) * limit
    return items[start:start + limit]


def page_count(n_items, limit=DEFAULT_LIMIT):
    """Number of pages needed for ``n_items`` items."""
    return max(1, -(-n_items // min(limit, MAX_LIMIT)))
