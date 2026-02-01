class Item:
    def __init__(self, name):
        self.name = "".join(sorted(set(name)))

    def combine(self, other):
        combined_letters = set(self.name) | set(other.name)
        new_name = "".join(sorted(combined_letters))
        return Item(new_name)

    def use(self):
        effects = {
            "X": -1,
            "XX": -2,
            "Y": -1,
            "YY": -2,
            "Z": -1,
            "ZZ": -2,

            "XY": +2,
            "XZ": +1,
            "YZ": +1,

            "XYZ": +3,
        }
        return effects.get(self.name, 0)

    def __str__(self):
        return self.name
