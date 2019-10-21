# search.py
# ---------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

"""
In search.py, you will implement generic search algorithms which are called
by Pacman agents (in searchAgents.py).
"""

import util



class linkedlist: #(successor, action, stepCost)

    def __init__(self,state, pervious, action, now_cost):
        self.state = state
        self.pervious = pervious
        if not pervious:
            self.action = []
        else:
            self.action = pervious.action + [action]


        if not pervious:
            self.cost = now_cost
        else:
            self.cost = pervious.cost + now_cost


    def get_state(self):
        return self.state 

    def get_pervious(self):
        return self.pervious

    def get_action(self):
        return self.action

    def get_cost(self):
        return self.cost


class linkedlist_with_h:

    def __init__(self,state, pervious, action, now_cost, heuristic):
        self.state = state
        self.pervious = pervious
        if not pervious:
            self.action = []
        else:
            self.action = pervious.action + [action]


        if not pervious:
            self.cost = now_cost
        else:
            self.cost = pervious.cost + now_cost

        self.heuristic = heuristic


    def get_state(self):
        return self.state 

    def get_pervious(self):
        return self.pervious

    def get_action(self):
        return self.action

    def get_cost(self):
        return self.cost

    def get_heristic(self):
        return self.heristic






class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other
    maze, the sequence of moves will be incorrect, so only use this for tinyMaze
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s,s,w,s,w,w,s,w]


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first
    [2nd Edition: p 75, 3rd Edition: p 87]

    Your search algorithm needs to return a list of actions that reaches
    the goal.  Make sure to implement a graph search algorithm
    [2nd Edition: Fig. 3.18, 3rd Edition: Fig 3.7].

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """

    stack = util.Stack()
    visited = set()
    dummy = linkedlist(problem.getStartState(), None, None, 0)
    stack.push(dummy)
    
    while not stack.isEmpty():

        node = stack.pop()
        visited.add(node.get_state())

        if problem.isGoalState(node.get_state()):
            # while node != dummy:
            #     path.append(node.get_action())
            #     node = node.pervious

            # path.reverse()
            # print(path)

            return node.get_action()



        else:
            successors = problem.getSuccessors(node.get_state())

            for successor in successors:
                state = successor[0]
                action = successor[1]
                now_cost = successor[2]
                if state in visited:
                    continue
                stack.push(linkedlist(state, node, action, now_cost))


    util.raiseNotDefined()




def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first.
    [2nd Edition: p 73, 3rd Edition: p 82]
    """
    queue = util.Queue()
    visited = []
    dummy = linkedlist(problem.getStartState(), None, None, 0)
    queue.push(dummy)
    queued = []
    queued.append(problem.getStartState())

    while not queue.isEmpty():

        node = queue.pop()
        visited.append(node.get_state())
        queued.remove(node.get_state())

        if problem.isGoalState(node.get_state()):
            
            return node.get_action()


        else:
            successors = problem.getSuccessors(node.get_state())

            for successor in successors:
                state = successor[0]
                action = successor[1]
                now_cost = successor[2]
                if state in visited or state in queued:
                    continue
                queue.push(linkedlist(state, node, action, now_cost))
                queued.append(state)


    util.raiseNotDefined()

def uniformCostSearch(problem):
    # "Search the node of least total cost first. "
    heap = util.PriorityQueue()
    visited = set()
    dummy = linkedlist(problem.getStartState(), None, None, 0)
    heap.push(dummy,0)
    path= []
    
    while not heap.isEmpty():

        node = heap.pop()
        visited.add(node.get_state())

        if problem.isGoalState(node.get_state()):
            
            return node.get_action()


        else:
            successors = problem.getSuccessors(node.get_state())

            for successor in successors:
                state = successor[0]
                if state in visited:
                    continue

                action = successor[1]
                now_cost = successor[2]
                heap.push(linkedlist(state, node, action, now_cost), node.get_cost()+now_cost)

    
    

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    # "Search the node that has the lowest combined cost and heuristic first."
    heap = util.PriorityQueue()
    visited = []
    dummy = linkedlist_with_h(problem.getStartState(), None, None, 0,0)
    heap.push(dummy,0+0)
    path= []
    
    while not heap.isEmpty():

        node = heap.pop()
        visited.append(node.get_state())

        if problem.isGoalState(node.get_state()):
            
            return node.get_action()


        else:
            successors = problem.getSuccessors(node.get_state())
            

            for successor in successors:
                state = successor[0]
                if state in visited:
                    continue

                action = successor[1]
                now_cost = successor[2]
                heap.push(linkedlist_with_h(state, node, action, now_cost, heuristic), node.get_cost() + now_cost + heuristic(state,problem))

    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
