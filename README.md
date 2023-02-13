# DDQN_new
Double Deep Q-Learning for Optimal execution problem

# DDQN
Double Deep Q-Network for optimal execution
The file is made up of three scripts:
1) Ambiente:
  is the environment where the agent operates, gives the dynamics for the price and does useful transofrmations
  of the data in order to feed them into the Q-NN
2) Agente:
  is the agent with his decision rules, builds the Q-NN, chooses a policy, calculates the reward for 
  every episode. This happens within the train method defined therein.
3) Main:
  is the main function that calls the others and sets up the training and testing of the algorithm, returns the actions and the rewards.
