# -*- coding: utf-8 -*-
from collections import deque

# experiences replay buffer size
REPLAY_SIZE = 2000
# memory size 1000
# size of minibatch
small_BATCH_SIZE = 16
big_BATCH_SIZE = 128
BATCH_SIZE_door = 1000

# these are the hyper Parameters for DQN
# discount factor for target Q to caculate the TD aim value
GAMMA = 0.9
# the start value of epsilon E
INITIAL_EPSILON = 0.5
# the final value of epsilon
FINAL_EPSILON = 0.01


class DQN():

    def __init__(self, observation_width, observation_height, action_space,
                 model_file, log_file):
        # the state is the input vector of network, in this env, it has four dimensions
        self.state_dim = observation_width * observation_height
        self.state_w = observation_width
        self.state_h = observation_height
        # the action is the output vector and it has two dimensions
        self.action_dim = action_space
        # init experience replay, the deque is a list that first-in & first-out
        self.replay_buffer = deque()
        # set the value in choose_action
        self.epsilon = INITIAL_EPSILON
        self.model_path = model_file + "/save_model.pth"
        self.model_file = model_file
        self.log_file = log_file

    # this is the function that use the network output the action
    def Choose_Action(self, state):
        pass

    # this the function that store the data in replay memory
    def Store_Data(self, state, action, reward, next_state, done):
        pass

    # train the network, update the parameters of Q_value
    def Train_Network(self, BATCH_SIZE, num_step):
        pass

    def Update_Target_Network(self):
        pass

    def save_model(self):
        pass
