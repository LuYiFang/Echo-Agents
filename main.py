from Agent.CustomAgents.AggressiveAgent import AggressiveAgent
from Agent.CustomAgents.GenerousAgent import GenerousAgent
from Agent.CustomAgents.SilentAgent import SilentAgent


def play_round(agents, round_num):
    print(f"\n========== Round {round_num} ==========")

    print("\n[Receiving Phase]")
    for agent in agents:
        if agent.is_alive():
            agent.decide_receive()

    print("\n[Action Phase]")
    for agent in agents:
        if agent.is_alive():
            agent.hp -= 1
            if not agent.is_alive():
                print(f"\n--- {agent.name}'s Turn ---")
                print(f"{agent.name} has been eliminated (HP=0)")
                continue

            others = [a for a in agents if a != agent and a.is_alive()]

            print(f"\n--- {agent.name}'s Turn ---")
            target, speech = agent.decide_speech(others)
            agent.execute_speech(target, speech)

            action = agent.decide_action(others)
            agent.execute_action(action, others)

    print("\n[End of Round Status]")
    for agent in agents:
        items_str = ", ".join([str(agent.base_item)] + [str(i) for i in agent.items]) if agent.items or agent.base_item else "None"
        status = f"{agent.name:<8} | HP: {agent.hp:<3} | Items: {items_str}"
        if not agent.is_alive():
            status += "  (Dead)"
        print(status)

    print("======================================")


if __name__ == '__main__':
    agents = [
        AggressiveAgent("Agent1", "X"),
        GenerousAgent("Agent2", "Y"),
        SilentAgent("Agent3", "Z")
    ]

    round_num = 1
    while sum(a.is_alive() for a in agents) > 1:
        play_round(agents, round_num)
        round_num += 1

    alive_agents = [a for a in agents if a.is_alive()]
    if alive_agents:
        print(f"\nWinner: {alive_agents[0].name}")
    else:
        print("\nAll agents are dead. No winner.")
