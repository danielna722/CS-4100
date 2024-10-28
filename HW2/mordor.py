import numpy as np
from tqdm import tqdm
import gymnasium as gym
import matplotlib.pyplot as plt
import json 
import matrix_mdp
import sys


nS = 16
nA = 4

slip_prob = .1  # may want to test with different values

actions = ['up', 'down', 'left', 'right']  # Human readable labels for actions

p_0 = np.array([0 for _ in range(nS)])
p_0[12] = 1

P = np.zeros((nS,nS,nA), dtype=float)

def valid_neighbors(i,j):
    neighbors = {}
    if i>0:
        neighbors[0]=(i-1,j)
    if i<3:
        neighbors[1]=(i+1,j)
    if j>0:
        neighbors[2]=(i,j-1)
    if j<3:
        neighbors[3]=(i,j+1)
    return neighbors

for i in range(4):
    for j in range(4):
        if i==0 and j==2:
            continue            # outgoing probabilities from terminal states should be 0 in gymnasium
        if i==3 and j==1:
            continue            # outgoing probabilities from terminal states should be 0 in gymnasium

        neighbors = valid_neighbors(i,j)
        for a in range(nA):
            if a in neighbors:
                P[neighbors[a][0]*4+neighbors[a][1], i*4+j, a] = 1-slip_prob
                for b in neighbors:
                    if b != a:
                        P[neighbors[b][0]*4+neighbors[b][1], i*4+j, a] = slip_prob/float(len(neighbors.items())-1)

#################################################################
# REWARD MATRIX

# In this implementation, you only get the reward if you *intended* to get to 
# the target state with the corresponding action, but not through slipping.

# Doesn't really affect the implementation of your assignment questions below. 

#################################################################

R = np.zeros((nS, nS, nA))

R[2,1,3] = 2000
R[2,3,2] = 2000
R[2,6,0] = 2000

R[13,9,1] = 2
R[13,14,2] = 2
R[13,12,3] = 2

R[11,15,0] = -100
R[11,7,1] = -100
R[11,10,3] = -100
R[10,14,0] = -100
R[10,6,1] = -100
R[10,11,2] = -100
R[10,9,3] = -100
R[9,10,2] = -100
R[9,13,0] = -100
R[9,5,1] = -100
R[9,8,3] = -100

env=gym.make('matrix_mdp/MatrixMDP-v0', p_0=p_0, p=P, r=R)

#################################################################
# Helper Functions
#################################################################

#reverse map observations in 0-15 to (i,j)
def reverse_map(observation):
    return observation//4, observation%4

#################################################################
# Q-Learning
#################################################################

# STUDENTS TO IMPLEMENT THIS FUNCTION

'''

In this section, you will implement a function for Q-learning with epsilon-greedy exploration.
Refer to the written assignment for the update equation. Use the following code to take an action:

observation, reward, terminated, truncated, info = env.step(action)

Your action is now chosen by the epsilon-greedy policy. The action is chosen as follows:

With probability epsilon, choose a random legal action.
With probability (1 - epsilon), choose the action that maximizes the Q-value (based on the last estimate). 
In case of ties, choose the action with the smallest index.

In case the chosen action is not a legal move, generate a random legal action.

The episode terminates when the agent reaches one of two terminal states. 

The Q-table is initialized to all zeros. The learning rate will vary over time and for every (s,a) pair.
It should be initialized as 1 and each time Q(s, a) is updated, it should become 1/(1 + number of updates to Q(s,a)). 

The number of updates to Q(s,a) should be stored in a matrix of shape (nS, nA) initialized to zeros, 
and updated such that num_updates[s,a] gives you the number of times Q(s,a) has been updated.
You can then calculate eta using the formula above.

The value of epsilon should be decayed to (0.9999 * epsilon) at the end of each episode.

After 10, 500, and 10000 episodes, plot a heatmap of max Q(s, a) for all states s. Complete and use the plot_heatmaps() function. 
The heatmap should be a 4x4 grid, corresponding to our map of Mordor. Please use plt.savefig() to save the plot, and do not use plt.show().

'''

def valid_action(observation, action):
    for i in range(nS):
        if P[i, observation, action] > 0:
            return True
    return False

def q_learning(num_episodes, checkpoints, gamma=0.9, epsilon=0.9):
    """
    Q-learning algorithm.

    Parameters:
    - num_episodes (int): Number of Q-value episodes to perform.
    - checkpoints (list): List of episode numbers at which to record the optimal value function..

    Returns:
    - Q (numpy array): Q-values of shape (nS, nA) after all episodes.
    - optimal_policy (numpy array): Optimal policy, np array of shape (nS,), ordered by state index.
    - V_opt_checkpoint_values (list of numpy arrays): A list of optimal value function arrays at specified episode numbers.
      The saved values at each checkpoint should be of shape (nS,).
    """
    
    Q = np.zeros((nS, nA))
    num_updates = np.zeros((nS, nA))

    observation, info = env.reset()

    V_opt_checkpoint_values = []
    optimal_policy = np.zeros(nS)

    

    '''


    YOUR CODE HERE


    '''
    state = 0
    action = None
    # epsilon = randomly pick a move (exploration)
    # 1 - epsilon = choose optimal (exploitation)

    for i in tqdm(range(num_episodes)):
        observation, info = env.reset() # reset the env every episode
        while True:
            #choose random move
            if epsilon > np.random.rand():
                action = np.random.randint(nA)
                while not valid_action(observation, action):
                    action = np.random.randint(nA)
            else:   #choose optimal
                action = np.argmax(Q[observation])
                while not valid_action(observation, action):
                    action = np.random.randint(nA)
            
            new_observation, reward, terminated, truncated, info = env.step(action)
            num_updates[observation, action] += 1

            a = 1 / num_updates[observation, action]

            Q[observation, action] = Q[observation, action] + a * (reward + gamma * np.max(Q[new_observation]) - Q[observation, action])

            observation = new_observation

            if terminated:
                break

        epsilon = epsilon * 0.9999

        if i + 1 in checkpoints:
            V_opt_checkpoint_values.append(np.max(Q, axis=1))
        
        
    optimal_policy = np.argmax(Q, axis=1)
    return Q, optimal_policy, V_opt_checkpoint_values





def plot_heatmaps(V_opt, filename):
    """
    Plots a 4x4 heatmap of the optimal value function, with state positions 
    corresponding to cells in the map of Mordor, with the given filename.

    Do not use plt.show().

    Parameters:
    V_opt (numpy array): A numpy array of shape (nS,) representing the optimal value function.
    filename (str): The filename to save the plot to. 

    Returns:
    None
    """

    '''


    YOUR CODE HERE


    '''
    grid = V_opt.reshape(4, 4)
    plt.figure(figsize=(6, 4))
    plt.imshow(grid, cmap="hot", interpolation= "nearest")
    plt.colorbar()
    plt.title("Heatmap")
    plt.xlabel("Column")
    plt.ylabel("Row")
    plt.savefig(filename)
    plt.close()


'''
If you need to make changes below for debugging, please first note down the defaults specified below.
Your submission should include plots generated using these default values, and slip_prob=0.1 (set in line 13).
'''

def main(): 

    Q, optimal_policy, V_opt_checkpoint_values = q_learning(10000, checkpoints=[10, 500, 10000])
    plot_heatmaps(V_opt_checkpoint_values[0], "heatmap_10.png")
    plot_heatmaps(V_opt_checkpoint_values[1], "heatmap_500.png")
    plot_heatmaps(V_opt_checkpoint_values[2], "heatmap_10000.png")

    
    
if __name__ == "__main__":
    main()