# The Game of Poker

## Overview

This repository contains an implementation of Texas Hold'em poker using Python and the [PyPokerEngine](https://github.com/ishikota/PyPokerEngine) framework. It demonstrates different poker-playing strategies through customizable player agents. The latest version integrates AI-driven strategies using Minimax with alpha-beta pruning and Monte Carlo simulations.

## Features

* **Player Strategies:**

  * `raise_player`: Advanced strategy using Minimax and Monte Carlo simulations for decision making.
  * `randomplayer`: Makes random moves, suitable for baseline comparisons.

* **Minimax with Alpha-Beta Pruning:**

  * The agent constructs a game tree of possible moves and counter-moves.
  * Uses alpha-beta pruning to eliminate branches that do not influence the final decision, allowing deeper and faster searches.
  * Evaluates terminal game states using a heuristic based on hand strength and game context.

* **Monte Carlo Simulations:**

  * Runs thousands of simulated outcomes from the current game state.
  * Estimates the expected value (EV) of actions such as fold, call, and raise.
  * Allows decision making in the face of uncertainty, particularly useful with hidden cards and unknown outcomes.

* **Performance Evaluation:**

  * Scripts included for evaluating the agent’s effectiveness against baseline strategies.

* **Ease of Customization:**

  * Modular setup makes it easy to implement and test new strategies.

## Requirements

* Python 3.x
* NumPy
* PyPokerEngine

Install dependencies using:

```bash
pip install numpy PyPokerEngine
```

## Usage

Run an example game simulation with:

```bash
python example.py
```

This will simulate gameplay between different strategy agents.

Evaluate performance with:

```bash
python testperf.py
```

## Project Structure

```
.
├── pypokerengine/           # PyPokerEngine framework directory
├── report/                  # Reports and project documentation
├── example.py               # Script to run poker game simulation
├── raise_player.py          # Intelligent strategy player using Minimax and Monte Carlo
├── randomplayer.py          # Random strategy player
├── test_preprocess.py       # Data preprocessing tests
├── testperf.py              # Performance evaluation script
├── .gitignore               # Git ignore configurations
├── ignore.txt               # Additional ignore configurations
└── .DS_Store                # macOS system file
```

## Future Improvements

* Enhance heuristics in Minimax for more accurate state evaluation.
* Extend opponent modeling for dynamic strategy adaptation.
* Integrate a graphical user interface for interactive play and analysis.

## Author

* **Sahana Shetty** - [GitHub](https://github.com/sahanashetty11)

