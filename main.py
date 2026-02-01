from Agent.CustomAgents.AggressiveAgent import AggressiveAgent
from Agent.CustomAgents.GenerousAgent import GenerousAgent
from Agent.CustomAgents.SilentAgent import SilentAgent
from Game import Game

agents = [
    AggressiveAgent("Agent1", "X"),
    GenerousAgent("Agent2", "Y"),
    SilentAgent("Agent3", "Z")
]

game = Game(agents)
game.run()