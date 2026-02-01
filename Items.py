class Item:
    def __init__(self, name):
        self.name = "".join(sorted(set(name)))

    def combine(self, other):
        combined_letters = set(self.name) | set(other.name)
        new_name = "".join(sorted(combined_letters))
        return Item(new_name)

    def use(self):
        effects = {
            "XY": +2,
            "YZ": -2,
            "XZ": 0
        }
        return effects.get(self.name, 0)

    def __str__(self):
        return self.name