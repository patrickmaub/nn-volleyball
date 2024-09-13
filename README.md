# 2D Volleyball Game with Advanced AI

This project implements a 2D volleyball game where two AI agents learn to play against each other using Proximal Policy Optimization (PPO), a reinforcement learning algorithm.

## Features

- 2D volleyball game simulation using Pygame
- AI agents trained with PPO algorithm
- TensorBoard integration for monitoring training progress
- Adaptive difficulty: training stops when agents consistently achieve long rallies
- Watch mode to observe trained agents play

## Requirements

- Python 3.8+
- PyTorch
- Pygame
- NumPy
- TensorBoard

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/patrickmaub/2d-volleyball-ai.git
   cd 2d-volleyball-ai
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

To start training the AI agents:

```
python main.py