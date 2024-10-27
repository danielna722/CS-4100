# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"

    # Unlike BFS, DFS will use a Stack
    # This means that the most recently added nodes to the frontier will be explored (aka nodes of the higher level of the tree)

    frontier = util.Stack()
    visitedNodes = {}
    startingNode = Node(problem.getStartState(), [])
    frontier.push(startingNode)
    #check if the starting node has goal state
    if problem.isGoalState(startingNode.getState()):
        return []
    while frontier.isEmpty() == False:
        node = frontier.pop()
        for successorNode in node.getSuccessorNodes(problem):
            if successorNode.getState() not in visitedNodes.keys():
                if problem.isGoalState(successorNode.getState()):
                    return successorNode.getPreviousActions()
                else:
                    frontier.push(successorNode)
                    visitedNodes[successorNode.getState()] = 0 # path cost is not applicable for DFS!


    util.raiseNotDefined()






def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"

    #The only difference from DFS is that the frontier is a Queue instead of Stack
    #This means that the nodes that are added first (aka the nodes of the lower level of the tree) have priority to be explored
    frontier = util.Queue()
    visitedNodes = {}
    startingNode = Node(problem.getStartState(), [])
    frontier.push(startingNode)
    #check if the starting node has goal state
    if problem.isGoalState(startingNode.getState()):
        return []
    while frontier.isEmpty() == False:
        node = frontier.pop()
        for successorNode in node.getSuccessorNodes(problem):
            if successorNode.getState() not in visitedNodes.keys():
                if problem.isGoalState(successorNode.getState()):
                    return successorNode.getPreviousActions()
                else:
                    frontier.push(successorNode)
                    visitedNodes[successorNode.getState()] = 0 # path cost is not applicable for BFS!

    util.raiseNotDefined()

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"

    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"

    # For HW1, I used manhattenHeuristic
    # manhattanHeuristic calculates the distance from the current position to the goal state.
    frontier = util.PriorityQueue()
    visitedNodes = {}
    startingNode = Node(problem.getStartState(), [], 0, 0)
    frontier.push(startingNode, startingNode.getPathCost() + heuristic(startingNode.getState(), problem))
    #check if the starting node has goal state
    if problem.isGoalState(startingNode.getState()):
        return []
    while frontier.isEmpty() == False:
        node = frontier.pop()
        for successorNode in node.getSuccessorNodes(problem):
            if successorNode.getState() not in visitedNodes.keys():
                if problem.isGoalState(successorNode.getState()):
                    return successorNode.getPreviousActions()
                else:
                    frontier.push(successorNode, successorNode.getPathCost() 
                                  + heuristic(successorNode.getState(), problem))
                    visitedNodes[successorNode.getState()] = successorNode.getPathCost()

        



    util.raiseNotDefined()


class Node:
    

    def __init__(self, state, pActions, sCost = 1, pCost = 0, pNode = None):
        self.state = state
        self.parentNode = pNode
        self.previousActions = pActions
        self.stepCost= sCost
        self.pathCost= pCost
    
    def getState(self):
        return self.state
    
    def getParentNode(self):
        return self.parentNode
    
    def getPreviousActions(self):
        return self.previousActions

    def getStepCost(self):
        return self.stepCost

    def getPathCost(self):
        return self.pathCost   

    def getSuccessorNodes(self, problem):
        successorNodes = []
        for successor in problem.getSuccessors(self.getState()):
            state, action, sCost = successor[0], successor[1], successor[2]
            newNode = Node(state, self.getPreviousActions() + [action], sCost, self.getPathCost() + sCost, self)
            successorNodes.append(newNode)

        return successorNodes
        


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
