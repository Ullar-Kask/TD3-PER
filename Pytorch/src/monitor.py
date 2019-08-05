from collections import deque
import numpy as np

def train(env, agent, num_episodes=500, max_t=1000, learn_step=20):
    """ Monitor agent's performance.
    
    Params
    ======
    - env: instance of the environment
    - agent: instance of class Agent
    - num_episodes: number of episodes of agent-environment interaction
    - max_t: maximum number of timesteps per episode
    - learn_step: number of timesteps between invoking learning on agent
    - learn_batch: how many times to learn in a go
    
    Returns
    =======
    - scores: list containing received rewards
    """
    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    
    env_info = env.reset(train_mode=True)[brain_name]
    agent.reset()
    states = env_info.vector_observations
    while len(agent.memory) < agent.BUFFER_SIZE:
        actions = agent.act(states)
        env_info = env.step(actions)[brain_name]
        next_states = env_info.vector_observations
        rewards = env_info.rewards
        dones = env_info.local_done
        for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
            agent.step(state, action, reward, next_state, done)
        states = next_states
    
    # list containing scores from each episode
    episode_scores = []
    # last 100 scores
    scores_window = deque(maxlen=100)
    
    # for each episode
    for i_episode in range(1, num_episodes+1):
        # begin the episode
        env_info = env.reset(train_mode=True)[brain_name]
        num_agents = len(env_info.agents)
        agent.reset()
        
        # get the current state (for each agent)
        states = env_info.vector_observations
        
        # initialize the score (for each agent)
        scores = np.zeros(num_agents)
        
        for t in range(1, max_t+1):
            # agent selects an action
            actions = agent.act(states)
            
            # agent performs the selected action
            env_info = env.step(actions)[brain_name]
            
            # get the next state
            next_states = env_info.vector_observations
            
            # get the reward
            rewards = env_info.rewards
            
            # see if episode has finished
            dones = env_info.local_done
            
            # agent performs internal updates based on sampled experience
            for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
                agent.step(state, action, reward, next_state, done)
            
            # update the score
            scores += rewards
            
            # update the state (s <- s') to next time step
            states = next_states
            
            if t%learn_step == 0:
                agent.learn()
            
            if np.any(dones):
                break
        
        # save final score
        episode_mean_score = np.mean(scores)
        scores_window.append(episode_mean_score)
        episode_scores.append(episode_mean_score)
        mean_score = np.mean(scores_window)
        
        # monitor progress
        if i_episode % 10 == 0:
            print("\rEpisode {:d}/{:d} || Average score {:.2f}".format(i_episode, num_episodes, mean_score))
        
        # check if task is solved
        if i_episode >= 100 and mean_score >= 30.0:
            print('\nEnvironment solved in {:d} episodes. Average score: {:.2f}'.format(i_episode, mean_score))
            agent.save_weights()
            break
    if i_episode == num_episodes: 
        print("\nAgent stopped. Final score {:.2f}\n".format(mean_score))
    return episode_scores
