class Game:
    def __init__(self, agents):
        self.agents = agents
        self.round_num = 1

    def play_round(self):
        print(f"\n========== Round {self.round_num} ==========")

        print("\n[Receiving Phase]")
        for agent in self.agents:
            if agent.is_alive():
                agent.execute_receive()

        print("\n[Action Phase]")
        for agent in self.agents:
            if agent.is_alive():
                agent.hp -= 1
                if not agent.is_alive():
                    print(f"\n--- {agent.name}'s Turn ---")
                    print(f"{agent.name} has been eliminated (HP=0)")
                    continue

                others = [a for a in self.agents if a != agent and a.is_alive()]

                print(f"\n--- {agent.name}'s Turn ---")
                target, speech = agent.decide_speech(others)
                agent.execute_speech(target, speech)

                action = agent.decide_action(others)
                agent.execute_action(action, others)

        print("\n[End of Round Status]")
        for agent in self.agents:
            items_str = ", ".join([str(agent.base_item)] + [str(i) for i in agent.items]) if agent.items or agent.base_item else "None"
            status = f"{agent.name:<8} | HP: {agent.hp:<3} | Items: {items_str}"
            if not agent.is_alive():
                status += "  (Dead)"
            print(status)

        print("======================================")
        self.round_num += 1

    def run(self):
        while sum(a.is_alive() for a in self.agents) > 1:
            self.play_round()

        alive_agents = [a for a in self.agents if a.is_alive()]
        if alive_agents:
            print(f"\nWinner: {alive_agents[0].name}")
        else:
            print("\nAll agents are dead. No winner.")

        print("\n===== Summary of Agents =====")
        for agent in self.agents:
            print(f"\n{agent.name} history:")
            for round_num, kind, detail in agent.history:
                print(f"  Round {round_num}: {kind} → {detail}")
