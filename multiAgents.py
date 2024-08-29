# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util, math

from game import Agent

from functools import partial
from math import log


class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        legalMoves = gameState.getLegalActions()

        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.
        """
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]


        distanceToFood = [util.manhattanDistance(newPos, foodPos) for foodPos in newFood.asList()]
        closestDistanceToFood = min(distanceToFood) if distanceToFood else 1
        ghostDistances = [util.manhattanDistance(newPos, ghostState.getPosition()) for ghostState in newGhostStates]
        closestGhostDistance = min(ghostDistances) if ghostDistances else 1

        # change the score according to the distance to the closest food.
        foodScore = 1.0 / closestDistanceToFood
        risk = 0
        for dist, scaredTime in zip(ghostDistances, newScaredTimes):
            if dist <= 1 and scaredTime <= 1:
                risk -= 300
                # take a considerably large amount of points for being close to a non-scared ghost.
        remainingFoodScore = -len(newFood.asList())



        totalScore = successorGameState.getScore() + foodScore + risk + remainingFoodScore
        # calculating total score taking into consideration distance to food, risk of being close to ghost and food left.
        return totalScore


def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)



class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.
        """

        def minimax(state, depth, agentIndex):
            if state.isWin() or state.isLose() or depth == 0:
                return self.evaluationFunction(state)
            #maximizer
            if agentIndex == 0:
                return max(minimax(state.generateSuccessor(agentIndex, action), depth, 1) for action in
                           state.getLegalActions(agentIndex))
            #minimizer
            else:
                nextAgent = agentIndex + 1
                if nextAgent == state.getNumAgents():
                    nextAgent = 0
                    depth -= 1
                return min(minimax(state.generateSuccessor(agentIndex, action), depth, nextAgent) for action in
                           state.getLegalActions(agentIndex))


        correctMove = max(gameState.getLegalActions(0), key=lambda action:
        minimax(gameState.generateSuccessor(0, action), self.depth, 1))

        return correctMove


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """

        def maxValue(gameState, alpha, beta, depth):
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)
            v = -float('inf')


            for action in gameState.getLegalActions(0):
                v = max(v, minValue(gameState.generateSuccessor(0, action), alpha, beta, depth, 1))
                if v > beta:
                    return v

                alpha = max(alpha, v)
            return v

        def minValue(gameState, alpha, beta, depth, agentIndex):
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)
            v = float('inf')


            for action in gameState.getLegalActions(agentIndex):
                if agentIndex == gameState.getNumAgents() - 1:
                    v = min(v, maxValue(gameState.generateSuccessor(agentIndex, action), alpha, beta, depth + 1))
                else:
                    v = min(v, minValue(gameState.generateSuccessor(agentIndex, action), alpha, beta, depth,
                                        agentIndex + 1))

                if v < alpha:
                    return v

                beta = min(beta, v)
            return v


        alpha = -float('inf')
        beta = float('inf')
        correctMove = Directions.STOP
        value = -float('inf')


        for action in gameState.getLegalActions(0):
            nextState = gameState.generateSuccessor(0, action)
            nextValue = minValue(nextState, alpha, beta, 0, 1)


            if nextValue > value:
                value = nextValue
                correctMove = action
            alpha = max(alpha, value)

        return correctMove


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, game_state):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        moves, points = self.get_value(game_state, 0, 0)
        return moves
    def get_value(self, state, i, depth):
        if len(state.getLegalActions(i)) == 0 or depth == self.depth:

            return "", self.evaluationFunction(state)

        if i == 0:

            return self.max_value(state, i, depth)


        else:

            return self.expected_value(state, i, depth)
    def max_value(self, state, i, depth):
        legalMoves = state.getLegalActions(i)

        max_value = float("-inf")
        max_moves = ""


        for moves in legalMoves:
            successor = state.generateSuccessor(i, moves)

            successor_index = i + 1
            successor_depth = depth

            if successor_index == state.getNumAgents():
                successor_index = 0
                successor_depth = 1 + successor_depth
            current_moves, current_value = self.get_value(successor, successor_index, successor_depth)

            if current_value > max_value:
                max_value = current_value
                max_moves = moves

        return max_moves, max_value

    def expected_value(self, state, i, depth):
        legalMoves = state.getLegalActions(i)

        expected_value = 0
        expected_moves = ""
        successor_probability = 1.0 / len(legalMoves)

        for moves in legalMoves:
            successor = state.generateSuccessor(i, moves)
            successor_index = i + 1
            successor_depth = depth


            if successor_index == state.getNumAgents():
                successor_index = 0
                successor_depth = 1 + successor_depth
            current_moves, current_value = self.get_value(successor, successor_index, successor_depth)
            expected_value = expected_value + successor_probability * current_value


        return expected_moves, expected_value

def betterEvaluationFunction(currentGameState):
    """
    This function evaluates the game state based on multiple factors:
    - Proximity to the nearest food pellet.
    - Distance from ghosts and their scared state.
    - Number of food pellets remaining.
    - Current game score.
    The weights for these factors are adjusted to balance between aggressive food collection and ghost avoidance.
    """

    newPosition = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    points = currentGameState.getScore()
    DistanceToFood = [util.manhattanDistance(newPosition, foodPos) for foodPos in newFood.asList()]


    if DistanceToFood:
        points += 1.0 / min(DistanceToFood)


    for ghostState in newGhostStates:
        distance = util.manhattanDistance(newPosition, ghostState.getPosition())


        if ghostState.scaredTimer > 0:
            points += 50.0 / (distance + 1)
        else:
            if distance <= 1:
                points -= 50.0


    remainingFood = currentGameState.getNumFood()
    points -= 5.0 * remainingFood


    return points

# Abbreviation
better = betterEvaluationFunction
