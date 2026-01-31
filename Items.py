class Item:
    def __init__(self, name):
        self.name = name

    def combine(self, other):
        new_name = self.name + other.name
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
