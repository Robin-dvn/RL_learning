import gymnasium as gym
def policy(observation):
    player_sum,dealer_card,usable_ace = observation
    if player_sum >= 20:
        return 0
    else:
        return 1

def main():
    env = gym.make('Blackjack-v1', natural=False, sab=False,render_mode="human")
    observation, info = env.reset()

    for _ in range(10):
        action = policy(observation)  # agent policy that uses the observation and info
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset()

    env.close()



