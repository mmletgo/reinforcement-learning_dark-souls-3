# CS5446_project-reinforcement-learning_dark-souls-3

![image](/img/%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0%E7%8E%A9%E9%BB%91%E6%9A%97%E4%B9%8B%E9%AD%823.png)

This branch focuses on applying DQN approach to automate gameplay in Dark Souls 3. It leverages various code modules designed to facilitate actions such as game control, health monitoring, and model training.

## Directory Structure

### directkeys_dark3.py

    - Defines key mappings and action controls for the game

### find_blood_location.py

    - A utility to measure the health of the agent and the boss based on pixel calculations in the screen images.

### getkeys

    - 工具，用来获取人工按下的键位

### grabscreen.py

    - Captures screenshots to provide initial image inputs for the model.

### utility

    - 工具函数，血量计算，动作执行，游戏暂停

### setting.py

    - Configures parameters related to the screenshot window and the action space size.

### model_dqn1.py

    - Contains the definition and hyperparameter configurations of the DQN model used in the reinforcement learning process.

### resart.py

    - An auxiliary script that facilitates quick resets of the training process after the agent's death.

### reward_fc.py

    - Defines the reward function which is critical for training the agent.

### training.py

    - Outlines the complete training process for the reinforcement learning model.

### testing.py

    - Defines the testing process for the model, without involving parameter updates.
