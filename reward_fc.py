def action_judge(self_blood, next_self_blood, self_stamina, next_self_stamina,
                 boss_blood, next_boss_blood, action, prev_action, stop, emergence_break):
    # 设定体力阈值
    high_stamina_threshold = 50
    low_stamina_threshold = 20

    # 判断角色是否死亡
    if next_self_blood < 3:
        reward = -1000
        done = 1
        stop = 0
        emergence_break += 1 if emergence_break < 2 else 100
        return reward, done, stop, emergence_break
    # 判断 Boss 是否死亡
    elif next_boss_blood < 3:
        reward = 2000
        done = 0
        stop = 0
        emergence_break += 1 if emergence_break < 2 else 100
        return reward, done, stop, emergence_break
    else:
        # 初始化奖励
        self_blood_reward = 0
        boss_blood_reward = 0
        action_reward = 0
        stamina_penalty = 0
        idle_penalty = 0
        combo_reward = 0  # 用于奖励组合动作

        # 计算角色血量变化
        if next_self_blood < self_blood:
            self_blood_reward = (next_self_blood - self_blood) * 3  # 负值，受到伤害

        # 计算 Boss 血量变化
        if next_boss_blood < boss_blood:
            boss_blood_reward = (boss_blood - next_boss_blood) * 3  # 正值，给予更多奖励

        # 获取当前体力
        current_stamina = next_self_stamina

        # 动作奖励和体力管理
        if action in [1, 3]:  # 攻击动作
            if current_stamina > high_stamina_threshold:
                action_reward = 20  # 体力充足，鼓励攻击
            elif current_stamina < low_stamina_threshold:
                action_reward = -2  # 体力过低，惩罚攻击
            else:
                action_reward = 0  # 体力一般，不奖励也不惩罚

        elif action in [9, 10, 11, 12]:  # 翻滚动作
            if current_stamina > high_stamina_threshold:
                action_reward = 7  # 体力充足，适度奖励翻滚
            elif current_stamina < low_stamina_threshold:
                action_reward = -2  # 体力过低，惩罚翻滚
            else:
                action_reward = 0  # 体力一般，不奖励也不惩罚

        elif action in [5]:  # 向前移动动作
            action_reward = 2  # 鼓励向前移动

        elif action == 2:  # 防御动作
            action_reward = 15  # 鼓励防御

        elif action == 0:  # 无动作
            idle_penalty = -5  # 惩罚无所作为

        # 体力过低惩罚
        if current_stamina < low_stamina_threshold:
            stamina_penalty = -2  # 鼓励等待体力恢复

        # 组合动作奖励：如果上一个动作是向前翻滚或向前移动，当前动作是攻击，给予额外奖励
        if prev_action in [5, 9] and action in [1, 3]:
            combo_reward = 40  # 给予组合动作奖励

        # 总奖励计算
        reward = self_blood_reward + boss_blood_reward + action_reward + stamina_penalty + idle_penalty + combo_reward

        done = 0
        emergence_break = 0
        return reward, done, stop, emergence_break
