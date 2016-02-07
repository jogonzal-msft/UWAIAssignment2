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
import random, util

from game import Agent

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
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newFoodAsList = newFood.asList();
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        distanceToClosestFood = -1;
        distanceToClosestGhost = -1;
        numberOfFood = len(newFoodAsList);

        for ghost in newGhostStates:
            ghostPosition = ghost.configuration.pos;
            ghostDirection = ghost.configuration.direction;
            ghostScaredRemainingTime = ghost.scaredTimer;
            distanceToGhost = manhattanDistance(newPos, ghostPosition);
            canEatGhost = (ghostScaredRemainingTime > distanceToGhost);
            if ((not canEatGhost) and (distanceToClosestGhost == -1 or distanceToGhost < distanceToClosestGhost)):
                distanceToClosestGhost = distanceToGhost;

        for food in newFoodAsList:
            distanceToFood = manhattanDistance(food, newPos);
            if (distanceToClosestFood == -1 or distanceToFood < distanceToClosestFood):
                distanceToClosestFood = distanceToFood;

        # For reference, see the past score
        pastScore = successorGameState.getScore()

        closeGhostScore = 0;

        if (distanceToClosestGhost < 4):
            closeGhostScore = (4 - distanceToClosestGhost) * 3000;

        score = - numberOfFood * 1000 - distanceToClosestFood - closeGhostScore;

        # print ('Score is ' + str(score))

        return score;

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

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)





class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def nextAgent(self, successor, depth, agentIndex, numAgents):
        agentIndex += 1;
        if (agentIndex >= numAgents):
            agentIndex = 0;
            depth += 1;

        if (depth == self.depth or successor.isWin() or successor.isLose()):
            score = self.evaluationFunction(successor);
            return (None, score);
        else:
            isPacman = agentIndex == 0;
            if (isPacman):
                return self.MaxValue(successor, depth, agentIndex, numAgents);
            else:
                return self.MinValue(successor, depth, agentIndex, numAgents);

    def MaxValue(self, state, depth, agentIndex, numAgents):
        legalActions = state.getLegalActions(agentIndex);
        value = -9999999999999999999;
        bestAction = None;
        for action in legalActions:
            successor = state.generateSuccessor(agentIndex, action);
            nextAgentValues = self.nextAgent(successor, depth, agentIndex, numAgents)[1];
            if (nextAgentValues > value):
                value = nextAgentValues;
                bestAction = action;
        return (bestAction, value);

    def MinValue(self, state, depth, agentIndex, numAgents):
        legalActions = state.getLegalActions(agentIndex);
        value = 9999999999999999999;
        bestAction = None;
        for action in legalActions:
            successor = state.generateSuccessor(agentIndex, action);
            nextAgentValues = self.nextAgent(successor, depth, agentIndex, numAgents)[1];
            if (nextAgentValues < value):
                value = nextAgentValues;
                bestAction = action;
        return (bestAction, value);

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """

        numAgents = gameState.getNumAgents();

        bestChoice = self.nextAgent(gameState, 0, -1, numAgents);
        return bestChoice[0];

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    Infinity = 9999999999999999999;

    def nextAgent(self, successor, depth, agentIndex, numAgents, alpha, beta):
        agentIndex += 1;
        if (agentIndex >= numAgents):
            agentIndex = 0;
            depth += 1;

        if (depth == self.depth or successor.isWin() or successor.isLose()):
            score = self.evaluationFunction(successor);
            return (None, score);
        else:
            isPacman = agentIndex == 0;
            if (isPacman):
                return self.MaxValue(successor, depth, agentIndex, numAgents, alpha, beta);
            else:
                return self.MinValue(successor, depth, agentIndex, numAgents, alpha, beta);

    def MaxValue(self, state, depth, agentIndex, numAgents, alpha, beta):
        legalActions = state.getLegalActions(agentIndex);
        value = -self.Infinity;
        bestAction = None;
        for action in legalActions:
            successor = state.generateSuccessor(agentIndex, action);
            nextAgentValues = self.nextAgent(successor, depth, agentIndex, numAgents, alpha, beta)[1];
            if (nextAgentValues > value):
                value = nextAgentValues;
                bestAction = action;
            #Alpha beta pruning shortcut
            if (value > beta):
                return (bestAction, value);
            alpha = max(alpha, value);
        return (bestAction, value);

    def MinValue(self, state, depth, agentIndex, numAgents, alpha, beta):
        legalActions = state.getLegalActions(agentIndex);
        value = self.Infinity;
        bestAction = None;
        for action in legalActions:
            successor = state.generateSuccessor(agentIndex, action);
            nextAgentValues = self.nextAgent(successor, depth, agentIndex, numAgents, alpha, beta)[1];
            if (nextAgentValues < value):
                value = nextAgentValues;
                bestAction = action;
            #Alpha beta pruning shortcut
            if (value < alpha):
                return (bestAction, value);
            beta = min(beta, value);
        return (bestAction, value);

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """

        numAgents = gameState.getNumAgents();

        bestChoice = self.nextAgent(gameState, 0, -1, numAgents, -self.Infinity, self.Infinity);
        return bestChoice[0];

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

