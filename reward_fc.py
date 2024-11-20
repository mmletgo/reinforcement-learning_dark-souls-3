def action_judge(boss_blood, next_boss_blood, self_blood, next_self_blood,
                 stop, emergence_break):
    # get action reward
    # emergence_break is used to break down training
    if next_self_blood < 1:  # self dead
        if emergence_break < 2:
            reward = -10
            done = 1
            stop = 0
            emergence_break += 1
            return reward, done, stop, emergence_break
        else:
            reward = -10
            done = 1
            stop = 0
            emergence_break = 100
            return reward, done, stop, emergence_break
    elif next_boss_blood < 1:  # boss dead
        if emergence_break < 2:
            reward = 20
            done = 0
            stop = 0
            emergence_break += 1
            return reward, done, stop, emergence_break
        else:
            reward = 20
            done = 0
            stop = 0
            emergence_break = 100
            return reward, done, stop, emergence_break
    else:
        self_blood_reward = 0
        boss_blood_reward = 0
        

        if next_self_blood - self_blood < -7:
            if stop == 0:
                self_blood_reward = -6
                stop = 1
                # 防止连续取帧时一直计算掉血
        else:
            stop = 0
        if next_boss_blood - boss_blood <= -3:
            boss_blood_reward = 4
        # print("self_blood_reward:    ",self_blood_reward)
        # print("boss_blood_reward:    ",boss_blood_reward)
        reward = self_blood_reward + boss_blood_reward 
        done = 0
        emergence_break = 0
        return reward, done, stop, emergence_break