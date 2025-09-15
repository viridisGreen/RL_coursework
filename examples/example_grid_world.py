import os
os.system("clear")
import sys
sys.path.append("..")
from src.grid_world import GridWorld
import random
import numpy as np
from examples.arguments import args

# Example usage:
if __name__ == "__main__":

    env = GridWorld()
    state = env.reset()
    env.render()

    discount = 0.9
    values = np.zeros(env.num_states)

    # Add policy: random policy
    if args.policy == "random":
        policy_matrix = np.array([
            [0.1, 0.6, 0.1, 0.1, 0.1],
            [0.1, 0.6, 0.1, 0.1, 0.1],
            [0.1, 0.6, 0.1, 0.1, 0.1],
            [0.1, 0.6, 0.1, 0.1, 0.1],
            [0.6, 0.1, 0.1, 0.1, 0.1],
            [0.1, 0.1, 0.6, 0.1, 0.1],
            [0.1, 0.1, 0.6, 0.1, 0.1],
            [0. , 0. , 0. , 0. , 0. ],
            [0. , 0. , 0. , 0. , 0. ],
            [0.6, 0.1, 0.1, 0.1, 0.1],
            [0.1, 0.1, 0.6, 0.1, 0.1],
            [0. , 0. , 0. , 0. , 0. ],
            [0.1, 0.1, 0.1, 0.1, 0.6],
            [0.1, 0.1, 0.1, 0.6, 0.1],
            [0.1, 0.1, 0.1, 0.6, 0.1]
        ])
    elif args.policy == "deterministic":
        policy_matrix = np.array([
            [0., 1., 0., 0., 0.],
            [0., 1., 0., 0., 0.],  
            [0., 1., 0., 0., 0.], 
            [0., 1., 0., 0., 0.],  
            [1., 0., 0., 0., 0.], 
            [0., 0., 1., 0., 0.],  
            [0., 0., 1., 0., 0.], 
            [0., 0., 0., 0., 0.],  
            [0., 0., 0., 0., 0.],  
            [1., 0., 0., 0., 0.],  
            [0., 0., 1., 0., 0.], 
            [0., 0., 0., 0., 0.],  
            [0., 0., 0., 0., 1.], 
            [0., 0., 0., 1., 0.],  
            [0., 0., 0., 1., 0.]
        ])
    else:
        raise ValueError("Invalid policy")

    env.add_policy(policy_matrix)
    env.render(save_path=f"{args.policy}_policy.png")

    for t in range(50):
        if (t != 49):
            env.render()
        else:
            env.render(save_path=f"{args.policy}_trajectory.png")

        x, y = env.agent_state
        state_id = x + y * env.env_size[0]

        action_index = np.random.choice(np.arange(len(env.action_space)), p=policy_matrix[state_id])
        action = env.action_space[action_index]
        next_state, reward, done, info = env.step(action)
        values[state_id] += discount**t * reward

        print(f"Step: {t}, Action: {action}, State: {next_state+(np.array([1,1]))}, Reward: {reward}, Done: {done}")

    
    # Add state values
    env.add_state_values(values)

    # Render the environment
    env.render(animation_interval=2, save_path=f"{args.policy}_values.png")