from Agent.CustomAgents.NNAgent import NNAgent
from Agent.CustomAgents.RLAgent import RLAgent
from Agent.CustomAgents.RandomAgent import RandomAgent
from Game import Game

agents = [
    RandomAgent("Random", "X"),
    RLAgent("RL", "Y"),
    NNAgent("NN", "Z")
]

game = Game(agents)
game.run()
