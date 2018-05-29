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
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        score = successorGameState.getScore()

        # HINTS:
        # Given currentGameState and successorGameState, determine if the next state is good / bad
        # Compute a numerical score for next state that will reflect this
        # Base score = successorGameState.getScore() - Line 77
        # Can increase / decrease this score depending on:
        #   new pacman position, ghost position, food position,
        #   distances to ghosts, distances to food
        # You can choose which features to use in your evaluation function
        # You can also put more weight to some features

        # get position of ghost
        ghostPos=successorGameState.getGhostPositions()[0]
        # obtain distance of pacman from ghost using manhattan distance
        pac_ghost_distance=manhattanDistance(newPos,ghostPos)
        pac_food_distance = []

        # using hint:
        # "You can also put more weight to some features"
        # adding more weight to food and ghost for better score reflection
        # wt=2.0 Question q1: 3/4, pacman gets stuck sometimes (or computer just can't keep up)
        # wt=5.0 Question q1: 3/4
        # wt=7.0 Question q1: 3/4
        # wt=9.0 Question q1: 3/4
        wt=10.0 # Question q1: 4/4

        # if pacman is near ghost, deduct to score
        # decreasing score based on ghost position and new pacman position
        if pac_ghost_distance:
            score=score-(wt/pac_ghost_distance)

        # determine existing foods near pacman by getting
        # manhattan distance between pacman and food
        for food in newFood.asList():
          pac_food_distance.append(manhattanDistance(newPos,food))

        # if pacman is near food/s, add to score with nearest food
        # increasing score based on food position and new pacman position
        if len(pac_food_distance):
            score=score+(wt/min(pac_food_distance)) 
        
        return score

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
        "*** YOUR CODE HERE ***"
        currentDepth = 0
        currentAgentIndex = self.index # agent's index
        action,score = self.value(gameState, currentAgentIndex, currentDepth)
        return action 


    # Note: always returns (action,score) pair
    def value(self, gameState, currentAgentIndex, currentDepth):
      # pass
      # Update the depth if the index of our pacman agent is already greater than or equal to the number of agents (if minimax have already assessed the possible moves of all adversarial agents in the current depth).
      if currentAgentIndex >= gameState.getNumAgents():
        currentAgentIndex = 0
        currentDepth += 1


      # Stop recursion when the predfined tree limit is reached or pacman already won or lost the game.
      if currentDepth == self.depth or gameState.isWin() or gameState.isLose():
          return ('',self.evaluationFunction(gameState))


      # If index of the agent is zero, that means it is our agent. Always get the max_value or the action which will give the highest score for our agent. However if it is not zero, it means it is an adversarial agent. For adversarial agent, assuming that it is a rational or it always chooses the optimal solution, always pick the min_value or the action that will give the worst effect (lowest score) to our pacman agent.
      if currentAgentIndex == 0:
          return self.max_value(gameState, currentAgentIndex, currentDepth)
      else:
          return self.min_value(gameState, currentAgentIndex, currentDepth)


      # Check when to update depth
      # check if currentDepth == self.depth
      #   if it is, stop recursion and return score of gameState based on self.evaluationFunction
      # check if gameState.isWin() or gameState.isLose()
      #   if it is, stop recursion and return score of gameState based on self.evaluationFunction
      # check whether currentAgentIndex is our pacman agent or ghost agent
      # if our agent: return max_value(....)
      # otherwise: return min_value(....)

    # Note: always returns (action,score) pair
    def max_value(self, gameState, currentAgentIndex, currentDepth):
      # pass

      # Getting the agent's action that will give the highest score. Loops over each legal action or actions that the agent can do during the game. Gets the value or score by calling self.value(...) for the agent's action and compares it to the current value, then stores as the new current value which ever of them is higher. Finally returns the highest value and its corresponding action.
      current_value = float('-inf')
      action = None
      actions = gameState.getLegalActions(currentAgentIndex)

      for a in actions:
        successor = gameState.generateSuccessor(currentAgentIndex, a)
        _,value = self.value(successor, currentAgentIndex+1, currentDepth)
        if value>current_value:
          current_value = value
          action = a

      return (action, current_value)

      # loop over each action available to current agent:
      # (hint: use gameState.getLegalActions(...) for this)
      #     use gameState.generateSuccessor to get nextGameState from action
      #     compute value of nextGameState by calling self.value
      #     compare value of nextGameState and current_value
      #     keep whichever value is bigger, and take note of the action too
      # return (action,current_value)

    # Note: always returns (action,score) pair
    def min_value(self, gameState, currentAgentIndex, currentDepth):
      # pass

      # Getting the agent's action that will give the lowest score. Loops over each legal action or actions that the agent can do during the game. Gets the value or score by calling self.value(...) for the agent's action and compares it to the current value, then stores as the new current value which ever of them is lower. Finally returns the lowest value and its corresponding action.
      current_value = float('inf')
      action = None
      actions = gameState.getLegalActions(currentAgentIndex)

      for a in actions:
        successor = gameState.generateSuccessor(currentAgentIndex, a)
        _,value = self.value(successor, currentAgentIndex+1, currentDepth)
        if value<current_value:
          current_value = value
          action = a

      return (action, current_value)


      # loop over each action available to current agent:
      # (hint: use gameState.getLegalActions(...) for this)
      #     use gameState.generateSuccessor to get nextGameState from action
      #     compute value of nextGameState by calling self.value
      #     compare value of nextGameState and current_value
      #     keep whichever value is smaller, and take note of the action too
      # return (action,current_value)



class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        currentDepth = 0
        currentAgentIndex = self.index # agent's index
        alpha = float('inf') * -1
        beta = float('inf')
        action,score = self.value(gameState, currentAgentIndex, currentDepth, alpha, beta)
        return action 

    # Note: always returns (action,score) pair
    def value(self, gameState, currentAgentIndex, currentDepth, alpha, beta):
      # pass
       # Update the depth if the index of our pacman agent is already greater than or equal to the number of agents (if minimax have already assessed the possible moves of all adversarial agents in the current depth).
      if currentAgentIndex >= gameState.getNumAgents():
        currentAgentIndex = 0
        currentDepth += 1


      # Stop recursion when the predfined tree limit is reached or pacman already won or lost the game.
      if currentDepth == self.depth or gameState.isWin() or gameState.isLose():
          return ('',self.evaluationFunction(gameState))


      # If index of the agent is zero, that means it is our agent. Always get the max_value or the action which will give the highest score for our agent. However if it is not zero, it means it is an adversarial agent. For adversarial agent, assuming that it is a rational or it always chooses the optimal solution, always pick the min_value or the action that will give the worst effect (lowest score) to our pacman agent. It also passes the alpha and beta values as parameters to the min_value and max_value functions. Alpha and beta are used for pruning parts of the search tree which will no longer be visited.

      if currentAgentIndex == 0:
          return self.max_value(gameState, currentAgentIndex, currentDepth, alpha, beta)
      else:
          return self.min_value(gameState, currentAgentIndex, currentDepth, alpha, beta)

      # More or less the same with MinimaxAgent's value() method
      # Just update the calls to max_value and min_value (should now include alpha, beta params)

    # Note: always returns (action,score) pair
    def max_value(self, gameState, currentAgentIndex, currentDepth, alpha, beta):
      # pass

      # Getting the agent's action that will give the highest score. Loops over each legal action or actions that the agent can do during the game. Gets the value or score by calling self.value(...) for the agent's action and compares it to the current value, then stores as the new current value which ever of them is higher. Finally returns the highest value and its corresponding action.
      current_value = float('-inf')
      action = None
      actions = gameState.getLegalActions(currentAgentIndex)

      if currentDepth == self.depth or gameState.isWin() or gameState.isLose():
          return ('',self.evaluationFunction(gameState))

      for a in actions:
        successor = gameState.generateSuccessor(currentAgentIndex, a)
        _,value = self.value(successor, currentAgentIndex+1, currentDepth, alpha, beta)
        if value>current_value:
          current_value = value
          action = a

        # If current value is worst than beta, immediately return the action and score, the remaining parts of the tree after the successor will be pruned. 
        if current_value> beta:
          return (action, current_value)

        # Updating alpha (best possible max so far)
        alpha = max(alpha, current_value)

      return (action, current_value)

      # Similar to MinimaxAgent's max_value() method
      # Include checking if current_value is worse than beta
      #   if so, immediately return current (action,current_value) tuple
      # Include updating of alpha

    # Note: always returns (action,score) pair
    def min_value(self, gameState, currentAgentIndex, currentDepth, alpha, beta):
      # pass

      # Getting the agent's action that will give the lowest score. Loops over each legal action or actions that the agent can do during the game. Gets the value or score by calling self.value(...) for the agent's action and compares it to the current value, then stores as the new current value which ever of them is lower. Finally returns the lowest value and its corresponding action.
      current_value = float('inf')
      action = None
      actions = gameState.getLegalActions(currentAgentIndex)

      if currentDepth == self.depth or gameState.isWin() or gameState.isLose():
          return ('',self.evaluationFunction(gameState))

      for a in actions:
        successor = gameState.generateSuccessor(currentAgentIndex, a)
        _,value = self.value(successor, currentAgentIndex+1, currentDepth, alpha, beta)
        if value<current_value:
          current_value = value
          action = a


        # If current value is worst than alpha, immediately return the action and score, the remaining parts of the tree after the successor will be pruned. 
        if current_value<   alpha:
          return (action, current_value)

        # Updating beta (best possible min so far)
        beta = min(beta, current_value)

      return (action, current_value)

      # Similar to MinimaxAgent's min_value() method
      # Include checking if current_value is worse than alpha
      #   if so, immediately return current (action,current_value) tuple
      # Include updating of beta

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
        currentDepth = 0
        currentAgentIndex = self.index # agent's index
        action,score = self.value(gameState, currentAgentIndex, currentDepth)
        return action 

    # Note: always returns (action,score) pair
    def value(self, gameState, currentAgentIndex, currentDepth):
      # pass

      # Update the depth if the index of our pacman agent is already greater than or equal to the number of agents (if minimax have already assessed the possible moves of all adversarial agents in the current depth).
      if currentAgentIndex >= gameState.getNumAgents():
        currentAgentIndex = 0
        currentDepth += 1


      # Stop recursion when the predfined tree limit is reached or pacman already won or lost the game.
      if currentDepth == self.depth or gameState.isWin() or gameState.isLose():
          return ('',self.evaluationFunction(gameState))

      # If index of the agent is zero, that means it is our agent. Always get the max_value or the action which will give the highest score for our agent. However if it is not zero, it means it is an adversarial agent. For adversarial agent, assuming that it is not a rational or not all the time it chooses the optimal solution, call exp_value which will compute for the average value based on the scores produced by the adversaries' actions.

      if currentAgentIndex == 0:
          return self.max_value(gameState, currentAgentIndex, currentDepth)
      else:
          return self.exp_value(gameState, currentAgentIndex, currentDepth)
      # More or less the same with MinimaxAgent's value() method
      # Only difference: use exp_value instead of min_value

    # Note: always returns (action,score) pair
    def max_value(self, gameState, currentAgentIndex, currentDepth):
      # pass

      # Getting the agent's action that will give the highest score. Loops over each legal action or actions that the agent can do during the game. Gets the value or score by calling self.value(...) for the agent's action and compares it to the current value, then stores as the new current value which ever of them is higher. Finally returns the highest value and its corresponding action.
      current_value = float('-inf')
      action = None
      actions = gameState.getLegalActions(currentAgentIndex)

      for a in actions:
        successor = gameState.generateSuccessor(currentAgentIndex, a)
        _,value = self.value(successor, currentAgentIndex+1, currentDepth)
        if value>current_value:
          current_value = value
          action = a

      return (action, current_value)
      # Exactly like MinimaxAgent's max_value() method

    # Note: always returns (action,score) pair
    def exp_value(self, gameState, currentAgentIndex, currentDepth):
      # pass

      # Getting the average value based on the values produced by all of the possible actions (legal actions) for the agent. Gets the value or score by calling self.value(...) for the agent's action, computes the probability of each action and adds the result to the current value. Finally returns the current value over the total number of actions, or the average value. None is returned for the action since we are only concerned with the average value and it is hard to determine the actual action to be taken.

      current_value = 0
      actions = gameState.getLegalActions(currentAgentIndex)
      probability = 1.0/len(actions)

      for a in actions:
        successor = gameState.generateSuccessor(currentAgentIndex, a)
        _,value = self.value(successor, currentAgentIndex+1, currentDepth)
        value *= probability
        current_value += value

      return (None, current_value)


      # use gameState.getLegalActions(...) to get list of actions
      # assume uniform probability of possible actions
      # compute probabilities of each action
      # be careful with division by zero
      # Compute the total expected value by:
      #   checking all actions
      #   for each action, compute the score the nextGameState will get
      #   multiply score by probability
      # Return (None,total_expected_value) 
      # None action --> we only need to compute exp_value but since the 
      # signature return values of these functions are (action,score), we will return an empty action


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    score = currentGameState.getScore()
    # Similar to Q1, only this time there's only one state (no nextGameState to compare it to)
    # Use similar features here: position, food, ghosts, scared ghosts, distances, etc.
    # Can use manhattanDistance() function
    # You can add weights to these features
    # Update the score variable (add / subtract), depending on the features and their weights
    # Note: Edit the Description in the string above to describe what you did here
    
    newScore = 0
    pacmanPosition = currentGameState.getPacmanPosition()
    ghostsPos = currentGameState.getGhostPositions(); #successor game state is not considered, i used the position of ghost from the current state replacing this-> ghostsPos = successorGameState.getGhostPositions()
    ghostScore = 0

    for ghostPos in ghostsPos:
        nearGhost = manhattanDistance(ghostPos, pacmanPosition)
        if(nearGhost < 2): #this tells us that pacman had a close encounter with the ghost
          ghostScore = 10000000000 * nearGhost #we set the weight of the distance of pacman to the ghost and multiplied it to the manhattan distance
    nearestFood = 1000
    curToNearestFood = 0
    for food in currentGameState.getFood().asList(): #evaluating the states of the food 
        nearFood = manhattanDistance(food, pacmanPosition)#getting the distance of the foods
        if(nearFood < nearestFood): 
            nearestFood = nearFood #taking note the distance of the nearest food
            curToNearestFood = manhattanDistance(food, pacmanPosition)

    if(ghostScore>newScore): #since the fact that pacman had a close encounter with the ghost is bad, it overrules all other features
      newScore = ghostScore
      newScore -= 10*currentGameState.getScore()
      
    else: #if pacman did not have a close encounter with the ghost, we use other important features as score instead
      newScore+=nearestFood
      newScore+= 1000*currentGameState.getNumFood()
      newScore+= 10*len(currentGameState.getCapsules())
      newScore-= 10*currentGameState.getScore()
    return newScore *(-1) #since we receprocate everything, it means that, the values we added more weight to are bad for pacman and the ones with less weight are good for pacman
    
    return score

# Abbreviation
better = betterEvaluationFunction

