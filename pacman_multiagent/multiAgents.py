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


import random
import sys
import util

from game import Agent


class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """

    def get_action(self, game_state):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        legal_moves = game_state.getLegalActions()

        scores = [self.evaluation_function(game_state, action)
                  for action in legal_moves]
        best_score = max(scores)
        best_indices = [index for index in range(len(scores))
                        if scores[index] == best_score]
        chosen_index = random.choice(best_indices)

        "Add more of your code here if you want to"

        return legal_moves[chosen_index]

    def evaluation_function(self, current_game_state, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (new_food) and Pacman position after moving (new_pos).
        new_scared_times holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """

        successor_game_state = current_game_state.generatePacmanSuccessor(action)

        new_pos = successor_game_state.getPacmanPosition()
        new_food = successor_game_state.getFood()
        new_ghost_states = successor_game_state.getGhostStates()
        new_scared_times = [ghostState.scaredTimer for ghostState in
                            new_ghost_states]
        walls = successor_game_state.getWalls().asList()

        score = successor_game_state.getScore()

        if successor_game_state.isWin(): return sys.maxsize
        if successor_game_state.isLose(): return -sys.maxsize

        food_list = new_food.asList()
        closest_food_dist = bfs_distance(new_pos, set(food_list), set(walls))
        score += 10.0 / (closest_food_dist + 1)

        ghost_positions = [ghost.getPosition() for ghost in new_ghost_states]
        closest_ghost_dist = bfs_distance(new_pos, set(ghost_positions), set(walls))

        if closest_ghost_dist < 2:
            score -= 50

        for ghost, scared_time in zip(new_ghost_states, new_scared_times):
            if scared_time > 0 and closest_ghost_dist < 2:
                score += 200

        return score


from collections import deque


def bfs_distance(start_pos, targets, walls):
    """
    Trouve la distance en nombre de pas depuis start_pos vers le point le plus proche parmi targets en évitant les murs.
    """
    if not targets:
        return float('inf')

    queue = deque([(start_pos, 0)])
    visited = set()

    while queue:
        (x, y), dist = queue.popleft()

        if (x, y) in visited:
            continue
        visited.add((x, y))

        if (x, y) in targets:
            return dist

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            next_pos = (x + dx, y + dy)
            if next_pos not in walls:
                queue.append((next_pos, dist + 1))

    return float('inf')


def score_evaluation_function(current_game_state):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search game
      (not reflex game).
    """
    return current_game_state.getScore()


class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search game.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn='score_evaluation_function', depth='2'):
        super().__init__()
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Implémente l'algorithme Minimax pour Pac-Man.
    """

    def get_action(self, game_state):
        """
        Retourne la meilleure action pour Pac-Man en utilisant Minimax.
        """

        def minimax(state, depth, agent_index):
            """
            Fonction récursive Minimax.
            """
            if depth == self.depth or state.isWin() or state.isLose():
                return self.evaluationFunction(state)

            if agent_index == 0:
                return max_value(state, depth)

            else:
                return min_value(state, depth, agent_index)

        def max_value(state, depth):
            """
            Fonction de maximisation pour Pac-Man.
            """
            best_score = float('-inf')
            best_action = None

            for action in state.getLegalActions(0):
                successor = state.generateSuccessor(0, action)
                score = minimax(successor, depth, 1)

                if score > best_score:
                    best_score = score
                    best_action = action

            if depth == 0:
                return best_action
            return best_score

        def min_value(state, depth, agent_index):
            """
            Fonction de minimisation pour les fantômes.
            """
            best_score = float('inf')
            next_agent = agent_index + 1
            num_agents = state.getNumAgents()

            if next_agent == num_agents:
                next_agent = 0
                depth += 1

            for action in state.getLegalActions(agent_index):
                successor = state.generateSuccessor(agent_index, action)
                score = minimax(successor, depth, next_agent)

                if score < best_score:
                    best_score = score

            return best_score

        return minimax(game_state, 0, 0)


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Implémente l'algorithme Minimax avec élagage alpha-bêta pour Pac-Man.
    """

    def get_action(self, game_state):
        """
        Retourne la meilleure action pour Pac-Man en utilisant Minimax avec élagage alpha-bêta.
        """

        def alphabeta(state, depth, agent_index, alpha, beta):
            """
            Fonction récursive Minimax avec élagage alpha-bêta.
            """
            if depth == self.depth or state.isWin() or state.isLose():
                return self.evaluationFunction(state)

            if agent_index == 0:
                return max_value(state, depth, alpha, beta)

            else:
                return min_value(state, depth, agent_index, alpha, beta)

        def max_value(state, depth, alpha, beta):
            """
            Fonction de maximisation pour Pac-Man.
            """
            best_score = float('-inf')
            best_action = None

            for action in state.getLegalActions(0):
                successor = state.generateSuccessor(0, action)
                score = alphabeta(successor, depth, 1, alpha, beta)

                if score > best_score:
                    best_score = score
                    best_action = action

                alpha = max(alpha, best_score)

                if best_score >= beta:
                    break

            return best_action if depth == 0 else best_score

        def min_value(state, depth, agent_index, alpha, beta):
            """
            Fonction de minimisation pour les fantômes.
            """
            best_score = float('inf')
            next_agent = agent_index + 1
            num_agents = state.getNumAgents()

            if next_agent == num_agents:
                next_agent = 0
                depth += 1

            for action in state.getLegalActions(agent_index):
                successor = state.generateSuccessor(agent_index, action)
                score = alphabeta(successor, depth, next_agent, alpha, beta)

                if score < best_score:
                    best_score = score

                beta = min(beta, best_score)

                if best_score <= alpha:
                    break

            return best_score

        return alphabeta(game_state, 0, 0, float('-inf'), float('inf'))


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Implémente l'algorithme Expectimax pour Pac-Man.
    """

    def get_action(self, game_state):
        """
        Retourne la meilleure action pour Pac-Man en utilisant l'algorithme Expectimax.
        """

        def expectimax(state, depth, agent_index):
            """
            Fonction récursive Expectimax.
            """
            if depth == self.depth or state.isWin() or state.isLose():
                return self.evaluationFunction(state)

            if agent_index == 0:
                return max_value(state, depth)

            else:
                return average_value(state, depth, agent_index)

        def max_value(state, depth):
            """
            Fonction de maximisation pour Pac-Man.
            """
            best_score = float('-inf')
            best_action = None

            for action in state.getLegalActions(0):
                successor = state.generateSuccessor(0, action)
                score = expectimax(successor, depth, 1)

                if score > best_score:
                    best_score = score
                    best_action = action

            return best_action if depth == 0 else best_score

        def average_value(state, depth, agent_index):
            """
            Fonction pour calculer la moyenne des scores.
            """
            next_agent = agent_index + 1
            num_agents = state.getNumAgents()

            if next_agent == num_agents:
                next_agent = 0
                depth += 1

            actions = state.getLegalActions(agent_index)
            if not actions:
                return self.evaluationFunction(state)

            total_score = 0
            for action in actions:
                successor = state.generateSuccessor(agent_index, action)
                total_score += expectimax(successor, depth, next_agent)

            return total_score / len(actions)

        return expectimax(game_state, 0, 0)


def betterEvaluationFunction(currentGameState):
    """
    Fonction d'évaluation améliorée pour Pac-Man.
    Elle prend en compte la nourriture, les fantômes, les capsules et la survie.
    """

    pacman_pos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood().asList()
    ghost_states = currentGameState.getGhostStates()
    scared_times = [ghost.scaredTimer for ghost in ghost_states]
    capsules = currentGameState.getCapsules()
    walls = currentGameState.getWalls().asList()

    score = currentGameState.getScore()

    if food:
        min_food_distance = bfs_distance(pacman_pos, set(food), set(walls))
        score += 10.0 / (min_food_distance + 1)

    for ghost, scared_time in zip(ghost_states, scared_times):
        ghost_pos = ghost.getPosition()
        ghost_distance = bfs_distance(pacman_pos, {ghost_pos}, set(walls))

        if scared_time > 0:
            if ghost_distance < 2:
                score += 200
        else:
            if ghost_distance < 2:
                score -= 300
            elif ghost_distance < 4:
                score -= 100

    score -= 50 * len(capsules)

    score -= 3 * len(food)

    if currentGameState.getPacmanState().getDirection() == "Stop":
        score -= 10

    for ghost, scared_time in zip(ghost_states, scared_times):
        ghost_pos = ghost.getPosition()
        ghost_distance = bfs_distance(pacman_pos, {ghost_pos}, set(walls))
        if scared_time > 0 and ghost_distance < 2:
            score += 200

    if capsules:
        nearest_capsule_distance = bfs_distance(pacman_pos, set(capsules), set(walls))
        score += 5.0 / (nearest_capsule_distance + 1)

    return score


better = betterEvaluationFunction