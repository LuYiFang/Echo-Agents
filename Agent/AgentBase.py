import random
from typing import final

from Items import Item


class Agent:
    def __init__(self, name, base_item_name):
        self.name = name
        self.base_item = Item(base_item_name)  # base item (infinite)
        self.items = []  # external items
        self.hp = 10
        self.incoming = []

        self.choices = ["accept", "reject"]
        self.speeches = ["", "A", "B", "AB"]
        self.actions = ["combine", "give", "use", "none"]


    @final
    def is_alive(self):
        """Check if agent is still alive (HP > 0)."""
        return self.hp > 0

    @final
    def execute_receive(self):
        """Execute receiving items based on decision."""
        for item, giver in self.incoming:
            decision = self.decide_receive(item, giver)
            if decision == "accept":
                self.items.append(item)
                print(f"{self.name} accepted {item} from {giver.name}")
            else:
                print(f"{self.name} rejected {item} from {giver.name}")
        self.incoming = []

    @final
    def execute_speech(self, target, speech):
        """Execute speech action."""
        if target and speech:
            print(f"{self.name} said '{speech}' to {target.name}")
        else:
            print(f"{self.name} said nothing")

    @final
    def execute_action(self, action, others):
        """Execute action (combine/give/use/none)."""
        if action == "combine" and self.items:
            item1 = self.base_item
            item2 = self.items.pop()
            new_item = item1.combine(item2)
            self.items.append(new_item)
            print(f"{self.name} combined {item1}+{item2} → created {new_item}")

        elif action == "give" and (self.items or self.base_item) and others:
            target = random.choice(others)
            if self.items and random.choice([True, False]):
                item = self.items.pop()
            else:
                item = Item(self.base_item.name)  # copy base item
            target.incoming.append((item, self))
            print(f"{self.name} tried to give {item} to {target.name}")

        elif action == "use" and self.items:
            item = self.items.pop()
            effect = item.use()
            self.hp += effect
            print(f"{self.name} used {item} → HP {effect:+}")

        else:
            print(f"{self.name} did nothing")

    # --- Decision methods (CAN override) ---
    def decide_receive(self, item, giver):
        """Decide whether to accept incoming items."""
        return random.choice(self.choices)

    def decide_speech(self, others):
        """Decide what to say and to whom."""
        speech = random.choice(self.speeches)
        target = random.choice(others) if speech and others else None
        return target, speech

    def decide_action(self, others):
        """Decide which action to take."""
        return random.choice(self.actions)
