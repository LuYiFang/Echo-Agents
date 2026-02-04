from Logger import Logger


class Game:
    def __init__(self, agents):
        self.agents = agents
        self.round_num = 1
        self.logger = Logger()

        for agent in self.agents:
            agent.logger = self.logger

    def play_round(self):
        self.logger.log(f"\n========== Round {self.round_num} ==========")

        self.logger.log("\n[Receiving Phase]")
        for agent in self.agents:
            if agent.is_alive():
                agent.execute_receive()

        self.logger.log("\n[Action Phase]")
        for agent in self.agents:
            if agent.is_alive():
                agent.hp -= 1
                if not agent.is_alive():
                    self.logger.log(f"\n--- {agent.name}'s Turn ---")
                    self.logger.log(f"{agent.name} has been eliminated (HP=0)")
                    continue

                others = [a for a in self.agents if a != agent and a.is_alive()]

                self.logger.log(f"\n--- {agent.name}'s Turn ---")
                target, speech = agent.decide_speech(others)
                agent.execute_speech(target, speech)

                action = agent.decide_action(others)
                agent.execute_action(action, others)

        self.logger.log("\n[End of Round Status]")
        for agent in self.agents:
            items_str = ", ".join([str(agent.base_item)] + [str(i) for i in agent.items]) if agent.items or agent.base_item else "None"
            status = f"{agent.name:<8} | HP: {agent.hp:<3} | Items: {items_str}"
            if not agent.is_alive():
                status += "  (Dead)"
            self.logger.log(status)

        self.logger.log("======================================")
        self.round_num += 1

    def run(self):
        while sum(a.is_alive() for a in self.agents) > 1:
            self.play_round()

        alive_agents = [a for a in self.agents if a.is_alive()]
        if alive_agents:
            self.logger.log(f"\nWinner: {alive_agents[0].name}")
        else:
            self.logger.log("\nAll agents are dead. No winner.")

        self.logger.save("game_log.txt")


