# Echo Agents

Echo Agents is a turn-based simulation where autonomous agents interact through **speaking**, **giving items**, **combining items**, and **using items**.  
Each agent shares the same set of possible actions, but their **decision-making logic** can differ, creating emergent and unpredictable gameplay.

---

## Game Rules

1. **Initial Setup**
   - Each agent starts with:
     - HP = 500
     - One infinite base item (cannot be consumed, can be copied to others)
     - An empty inventory for external items

2. **Round Structure**
   - **Receiving Phase**  
     Agents decide whether to accept or reject incoming items.
   - **Action Phase**  
     - Each alive agent loses 1 HP (endurance cost).  
     - If HP ≤ 0, the agent is eliminated immediately.  
     - Alive agents then:  
       a) Speak (say "A", "B", "AB", or nothing) to another agent.  
       b) Perform one action: `combine` / `give` / `use` / `none`.

3. **Actions**
   - **Combine**: base item + one external item → creates a new item (e.g., `XY`, `YZ`, `XZ`).  
   - **Give**: transfer one item (either a copy of base item or an external item) to another agent.  
   - **Use**: consume one item from inventory → affects HP:  
     - `XY` → +2 HP  
     - `YZ` → -2 HP  
     - `XZ` → 0 HP  
   - **None**: do nothing.

4. **Elimination**
   - Agents with HP ≤ 0 are out of the game and cannot act anymore.

5. **Victory**
   - The game continues until only one agent remains alive.  
   - That agent is declared the winner.

---

## Usage

Run the game:

```bash
python main.py
```

## Custom Agents

You can create custom agents by inheriting from the `Agent` class and implementing three main decision-making methods:

- `decide_receive(self, item, giver)` ➜ Decide whether to accept an item
- `decide_speech(self, others)` ➜ Decide what to say and to whom
- `decide_action(self, others)` ➜ Decide what action to perform