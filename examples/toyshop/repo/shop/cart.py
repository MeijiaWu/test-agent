"""Shopping cart."""


class Cart:
    def __init__(self):
        self.items = {}

    def add(self, sku, qty=1):
        if qty <= 0:
            raise ValueError("qty must be positive")
        self.items[sku] = self.items.get(sku, 0) + qty

    def total(self, prices):
        return sum(prices[sku] * qty for sku, qty in self.items.items())
