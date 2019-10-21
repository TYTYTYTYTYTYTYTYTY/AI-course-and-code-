# multiAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from util import manhattanDistance
from game import Directions
import random, util,sys

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
    #print(newPos)
    newFood = successorGameState.getFood()
    #print(newFood)
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    newGhostPosition = successorGameState.getGhostPositions()
    #print(newScaredTimes)
    score=successorGameState.getScore()

    foodPos = newFood.asList()
    foodDistance = []
    for i in range(len(foodPos)):
      foodDistance.append(manhattanDistance(newPos, foodPos[i]))
    numFood = len(foodDistance)

    GhostDistance=[]
    for i in range(len(newGhostStates)):
      GhostDistance.append(manhattanDistance(newPos, newGhostPosition[i]))

    cap=sum(newScaredTimes)

    if min(GhostDistance)<3:
      GhostPenalty=1
    else:
      GhostPenalty=0
    if foodDistance!=[]:
      score=1/(min(foodDistance)+0.1)-numFood-1000*GhostPenalty+cap+random.choice([0,0.5])+numFood/sum(foodDistance)
        #+random.choices([0,1],weights=(1-min(1,2*numFood/sum(GhostDistance)),min(1,2*numFood/sum(GhostDistance))))
    #current=currentGameState.getScore()

    "*** YOUR CODE HERE ***"
    return score

    
    #return successorGameState.getScore()

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


  def get_Successors(self,gameState, agentIndex):
    actions = gameState.getLegalActions(agentIndex)
    successors = [(action, gameState.generateSuccessor(agentIndex,action)) for action in actions if action != 'Stop']
    # if not successors:
    #     return[(Directions.STOP, gameState.generateSuccessor(agentIndex,Directions.STOP))]
    return successors


  def if_Terminal_state(self,gameState,depth):
    if gameState.isWin() or gameState.isLose() or self.depth == depth:
        return True
    else:
        return False


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

      Directions.STOP:
        The stop direction, which is always legal

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game
    """
    "*** YOUR CODE HERE ***"

    return self.max_step(gameState,0)
    util.raiseNotDefined()


  def max_step(self, gameState, depth):
    if self.if_Terminal_state(gameState,depth):
        return self.evaluationFunction(gameState)

    v = -sys.maxint

    for action, state in self.get_Successors(gameState,0):
        mini = self.mini_step(state,depth,state.getNumAgents()-1)
        if mini > v:
            max_Action = action
            v = mini

    if depth == 0 :
        return max_Action
    else:
        return v 
  

  def mini_step(self, gameState, depth, num_Ghosts):
    if self.if_Terminal_state(gameState,depth):
        return self.evaluationFunction(gameState)

    if num_Ghosts > 1:
        return min([self.mini_step(state,depth,num_Ghosts-1) for action,state in self.get_Successors(gameState,num_Ghosts)])
    if num_Ghosts == 1:
        return min([self.max_step(gameState,depth+1) for action,state in self.get_Successors(gameState,num_Ghosts)])






class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning (question 3)
  """

  def max_step_alphaBeta(self,gameState,depth,alpha,beta):
    if self.if_Terminal_state(gameState,depth):
        return self.evaluationFunction(gameState)

    v = -sys.maxint

    for action,state in self.get_Successors(gameState,0):
        mini = self.mini_step_alphaBeta(state,depth,gameState.getNumAgents()-1,alpha,beta)
        if mini > v:
            max_Action = action
            v = mini
        if v >=beta:
            return v 
        alpha = max(alpha,max_Action)

    if depth == 0:
        return max_Action
    else:
        return v





  def mini_step_alphaBeta(self,gameState,depth,num_Ghosts,alpha,beta):
    if self.if_Terminal_state(gameState,depth):
        return self.evaluationFunction(gameState)

    if num_Ghosts > 1:
        return min([self.mini_step_alphaBeta(state,depth,num_Ghosts-1 ,alpha,beta) for action,state in self.get_Successors(gameState,gameState.getNumAgents() - num_Ghosts) ])

    if num_Ghosts ==1:
        v = sys.maxint
        for action,state in self.get_Successors(gameState, gameState.getNumAgents()- num_Ghosts):
            v = min(v, self.max_step_alphaBeta(state,depth+1,alpha,beta))
            if v <= alpha:
                return v
            beta = min(beta,v)

        return v

  def getAction(self, gameState):
    """
      Returns the minimax action using self.depth and self.evaluationFunction
    """
    "*** YOUR CODE HERE ***"

    return self.max_step_alphaBeta(gameState,0,-sys.maxint,sys.maxint)
    

class ExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent (question 4)
  """


  def max_step_expMax(self, gameState,depth):
    if gameState.isWin() or gameState.isLose() or self.depth +1 == depth:
        return self.evaluationFunction(gameState)

    
    if depth == 0 :
        values  = [self.max_expected(state,depth, state.getNumAgents()-1) for action, state in self.get_Successors(gameState,0)]
        actions  = [action for action, state in self.get_Successors(gameState,0)]
        action = actions[[i for i,x in enumerate(values) if x == max(values)][0]]
        return action
    else:
        #return max([v]+ [self.max_expected(state,depth,state.getNumAgents()-1) for action, state in self.get_Successors(gameState,0)])
        v = -sys.maxint
        for action,state in self.get_Successors(gameState, 0):
            v = max([v, self.max_expected(state, depth, state.getNumAgents() - 1)])
        return v 
          

  def max_expected(self, gameState, depth, num_Ghosts):
    if self.if_Terminal_state(gameState,depth):
        return self.evaluationFunction(gameState)

    if num_Ghosts > 1:
        values  = [self.max_expected(state, depth, num_Ghosts -1) for action, state in self.get_Successors(gameState, gameState.getNumAgents() - num_Ghosts)]
    if num_Ghosts == 1:
        values = [self.max_step_expMax(state, depth+1) for action, state in self.get_Successors(gameState,gameState.getNumAgents() - num_Ghosts)]

    return sum(values)/len(values)


  def getAction(self, gameState):
    """
      Returns the expectimax action using self.depth and self.evaluationFunction

      All ghosts should be modeled as choosing uniformly at random from their
      legal moves.
    """
    "*** YOUR CODE HERE ***"
    return self.max_step_expMax(gameState,0)
    #util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
  """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
  """

  "*** YOUR CODE HERE ***"
  
  # Pos = currentGameState.getPacmanPosition()
  # Food = currentGameState.getFood()
  # GhostStates = currentGameState.getGhostStates()
  # ScaredTimes = [ghostState.scaredTimer for ghostState in  GhostStates]

  # # "*** YOUR CODE HERE ***"
  # if currentGameState.isWin():
  #     return sys.maxint
  # if currentGameState.isLose():
  #     return -sys.maxint

  # cloestGoastDistance = 1000000000
  # for ghost in GhostStates:
  #   if ghost.scaredTimer == 0:
  #     cloestGoastDistance = min(cloestGoastDistance, manhattanDistance(Pos, ghost.getPosition()))

  # foodList = Food.asList()
  # foodDistance = 0 
  # if foodList:
  #   distanceList = [manhattanDistance(Pos, foodPosition) for foodPosition in foodList]
  #   avgFoodDistance = sum(distanceList)/len(distanceList)
  #   minfoodDistance  = min(distanceList)
  #   maxfoodDistance = max(distanceList)

  # return currentGameState.getScore() + cloestGoastDistance* 1.6/minfoodDistance + 0.2*sum(ScaredTimes) + (cloestGoastDistance >=1 and cloestGoastDistance <3)*0.1 + (cloestGoastDistance >10) * 0.05
  successorGameState = currentGameState
  newPos = successorGameState.getPacmanPosition()
    #print(newPos)
  newFood = successorGameState.getFood()
    #print(newFood)
  newGhostStates = successorGameState.getGhostStates()
  newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

  newGhostPosition = successorGameState.getGhostPositions()
    #print(newScaredTimes)
  score=successorGameState.getScore()

  foodPos = newFood.asList()
  foodDistance = []
  for i in range(len(foodPos)):
    foodDistance.append(manhattanDistance(newPos, foodPos[i]))
  numFood = len(foodDistance)

  GhostDistance=[]
  for i in range(len(newGhostStates)):
    GhostDistance.append(manhattanDistance(newPos, newGhostPosition[i]))

  if min(GhostDistance)<=3:
    GhostPenalty=1
  else:
    GhostPenalty=0
    
  cap=sum(newScaredTimes)




  if foodDistance!=[]:
        #score=1/(min(foodDistance)+0.1)-numFood-1000*GhostPenalty+cap+random.choice([0,0.5])
    score=1.5/(min(foodDistance)+0.01)-numFood-10*GhostPenalty+0.15*cap+random.choice([0,0.5])+1/(max(foodDistance)+0.01)
  return score
    #util.raiseNotDefined()

  # if cloestGoastDistance > 3 :
  #   return 10000000 - 100*len(foodList) + 10/minfoodDistance  + *sum(ScaredTimes)+ random.choice([0,0.5])
  # else:
  #   return 10000000* cloestGoastDistance + 1/minfoodDistance

  


  #util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

class ContestAgent(MultiAgentSearchAgent):
  """
    Your agent for the mini-contest
  """

  def getAction(self, gameState):
    """
      Returns an action.  You can use any method you want and search to any depth you want.
      Just remember that the mini-contest is timed, so you have to trade off speed and computation.

      Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
      just make a beeline straight towards Pacman (or away from him if they're scared!)
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

