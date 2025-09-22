# Pac-Man AI Project

**Description:**
This project is a mini-laboratory assignment for the 4ALGL4A course where students implement rational agents to play Pac-Man. The goal is to explore artificial intelligence concepts such as reflex agents, adversarial search (Minimax), alpha-beta pruning, and Expectimax in a controlled Python 3 environment using Tkinter for the graphical interface.

---

## Repository Structure

```
.
├── pacman.py              # Main game script
├── graphicsDisplay.py     # GUI display
├── graphicsUtils.py       # GUI helper functions
├── textDisplay.py         # Text-based display
├── game.py                # Game state definitions
├── util.py                # Utility functions
├── projectParams.py       # Program parameters
├── layouts/               # Maze layout text files
├── layout.py              # Layout loader
├── keyboardAgents.py      # Human-controlled agent
├── ghostAgents.py         # Ghost AI agents
├── pacmanAgents.py        # Sample agents: LeftTurnAgent, GreedyAgent
├── multiAgents.py         # Your rational agents: ReflexAgent, MultiAgentSearchAgent, MinimaxAgent, AlphaBetaAgent, ExpectimaxAgent
```

---

## Agents Overview

1. **ReflexAgent** – Evaluates the immediate effect of each action and chooses the best.
2. **MultiAgentSearchAgent** – Base class for agents considering other agents' potential moves.
3. **MinimaxAgent** – Implements the Minimax algorithm for adversarial search.
4. **AlphaBetaAgent** – Optimized Minimax with alpha-beta pruning.
5. **ExpectimaxAgent** – Uses expected values for ghost actions instead of worst-case scenarios.

Each agent has two main methods: `get_action(self, game_state)` and `evaluation_function(self, current_game_state, action)`.

---

## Usage

1. Run the game with a human agent:

```bash
python pacman.py
```

2. Run with ReflexAgent:

```bash
python pacman.py -p ReflexAgent
```

3. Run with MinimaxAgent (depth 4):

```bash
python pacman.py -p MinimaxAgent -l minimaxClassic -a depth=4
```

4. Run with AlphaBetaAgent (depth 3):

```bash
python pacman.py -p AlphaBetaAgent -l smallClassic -a depth=3
```

5. Run with ExpectimaxAgent (depth 3):

```bash
python pacman.py -p ExpectimaxAgent -l minimaxClassic -a depth=3
```

Use `--help` to see all options:

```bash
python pacman.py --help
```

---

## Goals

* Implement rational agents for Pac-Man.
* Apply Minimax, Alpha-Beta pruning, and Expectimax algorithms.
* Improve evaluation functions to enhance agent performance.
* Test agents on various maze layouts and ghost configurations.

---

## References

* UC Berkeley Pac-Man AI project: [https://ai.berkeley.edu/multiagent.html](https://ai.berkeley.edu/multiagent.html)
* Minimax algorithm: [https://en.wikipedia.org/wiki/Minimax](https://en.wikipedia.org/wiki/Minimax)
* Alpha-beta pruning: [https://en.wikipedia.org/wiki/Alpha-beta\_pruning](https://en.wikipedia.org/wiki/Alpha-beta_pruning)



This repository is for educational purposes and course submission only.
