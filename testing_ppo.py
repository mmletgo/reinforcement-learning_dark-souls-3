# -*- coding: utf-8 -*-
import time
from model_ppo import PPO
from utility import pause_game, gamestatus
from setting import action_size, paused
import traceback
import datetime
# PPO算法的超参数
ppo_dict = {
    'gamma': 0.99,  # 折扣因子
    'lr': 1e-3,  # 学习率
    'eps_clip': 0.2,  # PPO中的剪辑阈值
    'K_epochs': 4  # 每次采样后更新策略的次数
}
EPISODES = 10  # 测试的最大次数

if __name__ == '__main__':
    agent = PPO(action_size, ppo_dict)
    newgame = gamestatus(2)
    paused = pause_game(paused)
    # paused at the begin
    emergence_break = 0
    # emergence_break is used to break down training
    # 用于防止出现意外紧急停止训练防止错误训练数据扰乱神经网络
    newgame.restart()
    for episode in range(EPISODES):
        states, self_blood, self_stamina, boss_blood = newgame.get_states_info(
        )
        if boss_blood < 100:
            print("can't find boss, restart")
            newgame.suicide_restart()
            newgame.restart()
            continue

        done = 0
        total_reward = 0
        stop = 0
        # 用于防止连续帧重复计算reward
        last_time = time.time()

        while True:
            last_time = time.time()
            # get the action by state
            try:
                action, logprob = agent.Choose_Action(states)
            except Exception:
                print("Choose_Action error")
                print(traceback.format_exc())
                break
            newgame.take_action(action)
            # take station then the station change
            next_states, next_self_blood, next_self_stamina, next_boss_blood = newgame.get_states_info(
            )
            reward, done, stop, emergence_break = newgame.action_judge(
                self_blood, next_self_blood, self_stamina, next_self_stamina,
                boss_blood, next_boss_blood, action, stop, emergence_break)
            # print(self_blood, next_self_blood, self_stamina, next_self_stamina,
            #       boss_blood, next_boss_blood, stop, emergence_break)
            # get action reward+we
            elapsed_time = time.time() - last_time
            if action != 0:
                current_time = datetime.datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S")
                print(
                    f'{current_time} - loop took {elapsed_time:.2f} s, action {newgame.action_dict[action]}, self_blood {self_blood}, next_self_blood {next_self_blood}, boss_blood {boss_blood}, next_boss_blood {next_boss_blood}, reward {reward}, done {done}'
                )
            if emergence_break == 100:
                # emergence break , save model and paused
                # 打倒了BOSS
                print("emergence_break")
                paused = True
                break

            agent.store_transition(states, action, logprob, reward)
            states = next_states
            self_blood = next_self_blood
            self_stamina = next_self_stamina
            boss_blood = next_boss_blood
            total_reward += reward
            paused = pause_game(paused)
            if done == 1:
                agent.clear_memory()
                newgame.reset()
                print(f'episode: {episode}, Reward:{total_reward}')
                break
        newgame.restart()

    agent.writer.close()
