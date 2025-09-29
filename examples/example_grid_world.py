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

    discount = args.discount_rate
    values = np.zeros(env.num_states)

    # Add policy: random policy
    if args.policy == "stochastic":
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

    args.policy_matrix = policy_matrix
    env.add_policy(policy_matrix)

    env.render(save_path=f"{args.policy}_policy-graph.png")
    env.get_r_and_p(policy_matrix, save_path=f"{args.policy}_r-and-P-values.png")


    for t in range(50):
        if (t != 49):
            env.render()
        else:
            env.render()  # assginment 2
            # env.render(save_path=f"trajectory_{args.policy}.png") # assginment 1

        x, y = env.agent_state
        state_id = x + y * env.env_size[0]

        action_index = np.random.choice(np.arange(len(env.action_space)), p=policy_matrix[state_id])
        action = env.action_space[action_index]
        next_state, reward, done, info = env.step(action)
        values[state_id] += discount**t * reward

        print(f"Step: {t}, Action: {action}, State: {next_state+(np.array([1,1]))}, Reward: {reward}, Done: {done}")

    
    # Add state values and render the environment
    env.add_state_values(
        policy_matrix, 
        gamma=discount,
        solution="closed", 
        save_path=f"{args.policy}_state-values-closed-form.png"
    )
    # env.render(animation_interval=2, save_path=f"{args.policy}_state-values-closed-form.png")

    env.add_state_values(
        policy_matrix, 
        gamma=discount,
        solution="iterative", 
        save_path=f"{args.policy}_state-values-iterative-form.png"
    )
    # env.render(animation_interval=2, save_path=f"{args.policy}_state-values-iterative-form.png")