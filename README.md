# Hybridised Temporal Difference Learning Ensemble with DQN Agents

This project implements a hybridised temporal difference learning ensemble using Deep Q-Network (DQN) agents. The ensemble approach combines model-free and model-based DQN agents to improve learning efficiency and performance in reinforcement learning tasks.

## Requirements

- Python 3.11+
- Poetry for dependency management

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/H1drogen/Hybridised-TemporalDifferenceWeighted-Ensemble.git
   cd Hybridised-TemporalDifferenceWeighted-Ensemble
   ```

2. Install dependencies with Poetry:
   ```bash
   poetry install
    ```

## Configuration

Edit the Environment settings and Hyperparameters in train_model.py

Edit Ensemble Hyperparameters in tdw/tdw_ensemble.py

Edit Agent Hyperparameters in DQN_Agent.py and DQN_Guided_Exploration.py


## Usage

3. To train the agent, run:
   ```bash
   poetry run python train_model.py
   ```

4. To run Evaluation, run:
   ```bash
   poetry run python evaluate_model.py
   ```
calling any evaluation metrics with the right path to datasets.