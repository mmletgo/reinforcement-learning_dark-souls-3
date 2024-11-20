# CS5446_project-reinforcement-learning_dark-souls-3

![image](/img/%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0%E7%8E%A9%E9%BB%91%E6%9A%97%E4%B9%8B%E9%AD%823.png)

This branch focuses on applying DQN approach to automate gameplay in Dark Souls 3. It leverages various code modules designed to facilitate actions such as game control, health monitoring, and model training.

本分支专注于应用 DQN 方法来实现《黑暗之魂 3》游戏玩法的自动化。它利用各种代码模块来促进游戏控制、健康监测和模型训练等操作。

## Directory Structure

### directkeys_dark3.py

    - Defines key mappings and action controls for the game
    - 键位映射以及动作定义

### find_blood_location.py

    - A utility to measure the health of the agent and the boss based on pixel calculations in the screen images.
    - 根据图像计算 agent 以及 boss 的血量

### getkeys.py
    - To capture key pressed by human
    - 工具，用来获取人工按下的键位

### grabscreen.py

    - Captures screenshots to provide initial image inputs for the model.
    - 截屏为模型提供初始的图像输入

### utility.py

    - Utility functions, including calculating blood for the agent and boss, taking actions and pause the game.
    - 工具函数，血量计算，动作执行，游戏暂停

### setting.py

    - Configures parameters related to the screenshot window and the action space size.
    - 关于截屏窗口的参数以及动作空间大小的配置

### model_dqn1.py

    - Contains the definition and hyperparameter configurations of the DQN model used in the reinforcement learning process.
    - 模型的定义以及超参数的配置

### resart.py

    - An auxiliary script that facilitates quick resets of the training process after the agent's death.
    - agent 死亡后快速重新新一轮训练的辅助脚本

### reward_fc.py

    - Defines the reward function which is critical for training the agent.
    - 奖励函数的定义

### training.py

    - Outlines the complete training process for the reinforcement learning model.
    - 定义了完整的训练过程

### testing.py

    - Defines the testing process for the model, without involving parameter updates.
    - 定义了测试过程,这个文件不涉及参数更新
