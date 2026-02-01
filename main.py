from Agent.CustomAgents.NNAgent import (AggressiveNNAgent,
                                        GenerousNNAgent, ConservativeNNAgent)
from Game import Game

agents = [
    AggressiveNNAgent("Aggressive", "X"),
    GenerousNNAgent("Generous", "Y"),
    ConservativeNNAgent("Conservative", "Z")
]

game = Game(agents)
game.run()
