from Agent.CustomAgents.NNAgent import NNAgent
from Agent.CustomAgents.QLearningAgent import QLearningAgent
from Agent.CustomAgents.RandomAgent import RandomAgent
from Game import Game

agents = [
    RandomAgent("Random", "X"),
    QLearningAgent("RL", "Y"),
    NNAgent("NN", "Z")
]

game = Game(agents)
game.run()
