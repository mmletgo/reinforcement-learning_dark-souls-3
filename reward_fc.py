def action_judge(self_blood, next_self_blood, self_stamina, next_self_stamina,
                 boss_blood, next_boss_blood, action, prev_action, stop, emergence_break):
    high_stamina_threshold = 50
    low_stamina_threshold = 20

    if next_self_blood < 3:
        reward = -1000
        done = 1
        stop = 0
        emergence_break += 1 if emergence_break < 2 else 100
        return reward, done, stop, emergence_break
    elif next_boss_blood < 3:
        reward = 2000
        done = 0
        stop = 0
        emergence_break += 1 if emergence_break < 2 else 100
        return reward, done, stop, emergence_break
    else:
        self_blood_reward = 0
        boss_blood_reward = 0
        action_reward = 0
        stamina_penalty = 0
        idle_penalty = 0
        combo_reward = 0

        if next_self_blood < self_blood:
            self_blood_reward = (next_self_blood - self_blood) * 3

        if next_boss_blood < boss_blood:
            boss_blood_reward = (boss_blood - next_boss_blood) * 3

        current_stamina = next_self_stamina

        if action in [1, 3]:
            if current_stamina > high_stamina_threshold:
                action_reward = 20
            elif current_stamina < low_stamina_threshold:
                action_reward = -2 
            else:
                action_reward = 0

        elif action in [9, 10, 11, 12]: 
            if current_stamina > high_stamina_threshold:
                action_reward = 7 
            elif current_stamina < low_stamina_threshold:
                action_reward = -2
            else:
                action_reward = 0

        elif action in [5]:
            action_reward = 2

        elif action == 2:
            action_reward = 15

        elif action == 0:
            idle_penalty = -5


        if current_stamina < low_stamina_threshold:
            stamina_penalty = -2

        if prev_action in [5, 9] and action in [1, 3]:
            combo_reward = 40
        reward = self_blood_reward + boss_blood_reward + action_reward + stamina_penalty + idle_penalty + combo_reward

        done = 0
        emergence_break = 0
        return reward, done, stop, emergence_break
