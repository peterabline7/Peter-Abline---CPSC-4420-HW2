import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import imageio

env = gym.make("Taxi-v4")

def train_model(alpha = 0.1, gamma = 0.99, eps = 1.0, eps_min = 0.05, eps_decay = 0.999, num_episodes = 300):
    Q = np.zeros((500, 6))
    all_rewards = []
    
    for episode in range(num_episodes):
        state, info = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            # epsilon-greedy: explore with probability epsilon, otherwise pick best action
            if np.random.rand() < eps:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state])
 
            next_state, reward, terminated, truncated, info = env.step(action)
 
            # terminal handling: if the episode ended, there is no future reward
            if terminated:
                target = reward
            else:
                target = reward + gamma * np.max(Q[next_state])
 
            # Q-learning update
            Q[state, action] = Q[state, action] + alpha * (target - Q[state, action])
 
            state = next_state
            total_reward = total_reward + reward
            done = terminated or truncated
 
        # shrink epsilon after each episode (but not below epsilon_min)
        eps = max(eps_min, eps * eps_decay)
        all_rewards.append(total_reward)
 
    return Q, all_rewards
 
 
def moving_average(rewards):
    # average of each reward with the 99 before it (smooths the plot)
    averages = []
    for i in range(100, len(rewards)):
        averages.append(np.mean(rewards[i - 100:i]))
    return averages
 
 
def evaluate(Q, num_episodes):
    # run the greedy policy (no exploration) and record results
    rewards = []
    steps = []
    successes = 0
    for episode in range(num_episodes):
        state, info = env.reset()
        done = False
        total_reward = 0
        num_steps = 0
        while not done:
            action = np.argmax(Q[state])
            state, reward, terminated, truncated, info = env.step(action)
            total_reward = total_reward + reward
            num_steps = num_steps + 1
            done = terminated or truncated
        if terminated:  # delivered the passenger
            successes = successes + 1
        rewards.append(total_reward)
        steps.append(num_steps)
    return np.mean(rewards), np.mean(steps), successes / num_episodes
 

# Baseline settings
ALPHA = 0.1
GAMMA = 0.99
EPSILON = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.999
EPISODES = 5000
 
Q, rewards = train_model(ALPHA, GAMMA, EPSILON, EPSILON_MIN, EPSILON_DECAY, EPISODES)
plt.figure()
plt.plot(moving_average(rewards))
plt.xlabel("Episode")
plt.ylabel("Reward (100-episode average)")
plt.title("Baseline")
plt.savefig("baseline.png")
 
# Part D: different learning rates
plt.figure()
for a in [0.1, 0.5, 0.9]:
    Q_a, r = train_model(a, GAMMA, EPSILON, EPSILON_MIN, EPSILON_DECAY, EPISODES)
    plt.plot(moving_average(r), label="alpha=" + str(a))
plt.xlabel("Episode")
plt.ylabel("Reward (100-episode average)")
plt.legend()
plt.savefig("alpha.png")
 
# Part E: different discount factors
plt.figure()
for g in [0.5, 0.9, 0.99]:
    Q_g, r = train_model(ALPHA, g, EPSILON, EPSILON_MIN, EPSILON_DECAY, EPISODES)
    plt.plot(moving_average(r), label="gamma=" + str(g))
plt.xlabel("Episode")
plt.ylabel("Reward (100-episode average)")
plt.legend()
plt.savefig("gamma.png")
 
# Part F: different epsilon decay rates 
plt.figure()
for d in [0.99, 0.999, 0.9995]:
    Q_d, r = train_model(ALPHA, GAMMA, EPSILON, EPSILON_MIN, d, EPISODES)
    plt.plot(moving_average(r), label="epsilon_decay=" + str(d))
plt.xlabel("Episode")
plt.ylabel("Reward (100-episode average)")
plt.legend()
plt.savefig("epsilon_decay.png")
 
#  Part G: evaluate the baseline Q-table with no exploration
mean_reward, mean_steps, success_rate = evaluate(Q, 30)
print("Mean reward:", mean_reward)
print("Mean steps:", mean_steps)
print("Success rate:", success_rate)




video_env = gym.make("Taxi-v4", render_mode="rgb_array")
frames = []
for episode in range(3):
    state, info = video_env.reset()
    frames.append(video_env.render())
    done = False
    while not done:
        action = np.argmax(Q[state])
        state, reward, terminated, truncated, info = video_env.step(action)
        frames.append(video_env.render())
        done = terminated or truncated

imageio.mimsave("taxi_agent.mp4", frames, fps=4)
print("saved taxi_agent.mp4")