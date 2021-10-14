# Deep Q-Learning

Deep Q Learning algorithm that learns to operate in one of the included environments:
- Cart Pole: classic environment from OpenAI Gym. A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The system is controlled by applying a force of +1 or -1 to the cart. The pendulum starts upright, and the goal is to prevent it from falling over. A reward of +1 is provided for every timestep that the pole remains upright. The episode ends when the pole is more than 15 degrees from vertical, or the cart moves more than 2.4 units from the center.out the shortest path across a warehouse.
    - https://gym.openai.com/envs/CartPole-v1/
- Warehouse: the agent moves across a warehouse finding the shortest path to an exit.

The projects builds from projects NN and QLearning, and aims to create a Reinforcement Learning algorithm from scratch that uses Deep Neural Networks.
- NN: basic neural network implementation.
- QLearning: simple Q-Learning implementation where the agent uses a Q Table to learn.

Current implementation uses 2 Deep Neural Networks (Main and Target) that update via Bellman's Equation and an Adam Optimizer.


## Usage

Upon starting program displays a simple menu:
1. Trains the model for a given number of episodes.
2. Uses the trained networks to simulate 10 episodes, rendering the ressult with OpenCV.
3. Saves the trained model.
4. Loads the trained model.
