__credits__ = ["Intelligent Unmanned Systems Laboratory at Westlake University."]

import sys    
sys.path.append("..")         
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches          
from examples.arguments import args           

class GridWorld():

    def __init__(self, env_size=args.env_size, 
                 start_state=args.start_state, 
                 target_state=args.target_state, 
                 forbidden_states=args.forbidden_states):

        self.env_size = env_size
        self.num_states = env_size[0] * env_size[1]
        self.start_state = start_state
        self.target_state = target_state
        self.forbidden_states = forbidden_states

        self.agent_state = start_state
        self.action_space = args.action_space          
        self.reward_target = args.reward_target
        self.reward_forbidden = args.reward_forbidden
        self.reward_step = args.reward_step

        self.canvas = None
        self.animation_interval = args.animation_interval


        self.color_forbid = (0.9290,0.6940,0.125)
        self.color_target = (0.3010,0.7450,0.9330)
        self.color_policy = (0.4660,0.6740,0.1880)
        self.color_trajectory = (0, 1, 0)
        self.color_agent = (0,0,1)

        # New
        self.num_valid_states = self.num_states - len(self.forbidden_states)


    def reset(self):
        self.agent_state = self.start_state
        self.traj = [self.agent_state] 
        return self.agent_state, {}


    def step(self, action):
        assert action in self.action_space, "Invalid action"

        next_state, reward  = self._get_next_state_and_reward(self.agent_state, action)
        done = self._is_done(next_state)

        x_store = next_state[0] + 0.03 * np.random.randn()
        y_store = next_state[1] + 0.03 * np.random.randn()
        state_store = tuple(np.array((x_store,  y_store)) + 0.2 * np.array(action))
        state_store_2 = (next_state[0], next_state[1])

        self.agent_state = next_state

        self.traj.append(state_store)   
        self.traj.append(state_store_2)
        return self.agent_state, reward, done, {}   
    
        
    def _get_next_state_and_reward(self, state, action):
        x, y = state
        new_state = tuple(np.array(state) + np.array(action))
        if y + 1 > self.env_size[1] - 1 and action == (0,1):    # down
            y = self.env_size[1] - 1
            reward = self.reward_forbidden  
        elif x + 1 > self.env_size[0] - 1 and action == (1,0):  # right
            x = self.env_size[0] - 1
            reward = self.reward_forbidden  
        elif y - 1 < 0 and action == (0,-1):   # up
            y = 0
            reward = self.reward_forbidden  
        elif x - 1 < 0 and action == (-1, 0):  # left
            x = 0
            reward = self.reward_forbidden 
        elif new_state == self.target_state:  # stay
            x, y = self.target_state
            reward = self.reward_target
        elif new_state in self.forbidden_states:  # stay
            x, y = state
            reward = self.reward_forbidden        
        else:
            x, y = new_state
            reward = self.reward_step
            
        return (x, y), reward
        

    def _is_done(self, state):
        return state == self.target_state
    

    def render(self, animation_interval=args.animation_interval, save_path=None):
        if self.canvas is None:
            plt.ion()                             
            self.canvas, self.ax = plt.subplots()   
            self.ax.set_xlim(-0.5, self.env_size[0] - 0.5)
            self.ax.set_ylim(-0.5, self.env_size[1] - 0.5)
            self.ax.xaxis.set_ticks(np.arange(-0.5, self.env_size[0], 1))     
            self.ax.yaxis.set_ticks(np.arange(-0.5, self.env_size[1], 1))     
            self.ax.grid(True, linestyle="-", color="gray", linewidth="1", axis='both')          
            self.ax.set_aspect('equal')
            self.ax.invert_yaxis()                           
            self.ax.xaxis.set_ticks_position('top')           
            
            idx_labels_x = [i for i in range(self.env_size[0])]
            idx_labels_y = [i for i in range(self.env_size[1])]
            for lb in idx_labels_x:
                self.ax.text(lb, -0.75, str(lb+1), size=10, ha='center', va='center', color='black')           
            for lb in idx_labels_y:
                self.ax.text(-0.75, lb, str(lb+1), size=10, ha='center', va='center', color='black')
            self.ax.tick_params(bottom=False, left=False, right=False, top=False, labelbottom=False, labelleft=False,labeltop=False)   

            self.target_rect = patches.Rectangle( (self.target_state[0]-0.5, self.target_state[1]-0.5), 1, 1, linewidth=1, edgecolor=self.color_target, facecolor=self.color_target)
            self.ax.add_patch(self.target_rect)     

            for forbidden_state in self.forbidden_states:
                rect = patches.Rectangle((forbidden_state[0]-0.5, forbidden_state[1]-0.5), 1, 1, linewidth=1, edgecolor=self.color_forbid, facecolor=self.color_forbid)
                self.ax.add_patch(rect)

            self.agent_star, = self.ax.plot([], [], marker = '*', color=self.color_agent, markersize=20, linewidth=0.5) 
            self.traj_obj, = self.ax.plot([], [], color=self.color_trajectory, linewidth=0.5)

        # self.agent_circle.center = (self.agent_state[0], self.agent_state[1])
        self.agent_star.set_data([self.agent_state[0]],[self.agent_state[1]])       
        traj_x, traj_y = zip(*self.traj)         
        self.traj_obj.set_data(traj_x, traj_y)

        plt.draw()
        plt.pause(animation_interval)
        if save_path is not None:
            plt.savefig(save_path)
            print(f"Rendered image saved to {save_path}")
        if args.debug:
            input('press Enter to continue...')     


    def draw_policy_table(self, policy_matrix, save_path=None):
        action_names = ["Action 1 (upward)", "Action 2 (rightward)", "Action 3 (downward)", "Action 4 (leftward)", "Action 5 (still)"]
        valid_rows = [(i, row) for i, row in enumerate(policy_matrix) if not np.all(row == 0)]
        
        indices, policies = zip(*valid_rows)
        row_labels = [f"State {i + 1}" for i in indices]
        table_data = [[f"{p:.2f}" for p in policy] for policy in policies]

        fig, ax = plt.subplots(figsize=(10, len(table_data) * 0.5))
        ax.axis('tight')
        ax.axis('off')
        
        the_table = ax.table(cellText=table_data,
                             rowLabels=row_labels,
                             colLabels=action_names,
                             loc='center',
                             cellLoc='center')
        
        the_table.auto_set_font_size(False)
        the_table.set_fontsize(10)
        the_table.scale(1.2, 1.2)
        plt.title(f"Policy Matrix: {args.policy}", fontsize=16)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)

        plt.close(fig)


    def get_r_and_p(self, policy_matrix, save_path=None):
        forbidden_grid_indices = {x + y * self.env_size[0] for x, y in self.forbidden_states}
        valid_to_grid_map = [idx for idx in range(self.num_states) if idx not in forbidden_grid_indices]
        num_valid_states = len(valid_to_grid_map)
        grid_to_valid_map = {grid_idx: valid_idx for valid_idx, grid_idx in enumerate(valid_to_grid_map)}

        r = np.zeros(num_valid_states)
        p = np.zeros((num_valid_states, num_valid_states))

        for valid_s_idx in range(num_valid_states):
            grid_s_idx = valid_to_grid_map[valid_s_idx]
            
            x = grid_s_idx % self.env_size[0]
            y = grid_s_idx // self.env_size[0]
            
            for a_idx, action in enumerate(self.action_space):
                action_prob = policy_matrix[grid_s_idx][a_idx]
                if action_prob == 0:
                    continue

                next_state_coords, reward = self._get_next_state_and_reward((x, y), action)
                grid_next_s_idx = next_state_coords[0] + next_state_coords[1] * self.env_size[0]
                valid_next_s_idx = grid_to_valid_map[grid_next_s_idx]

                r[valid_s_idx] += action_prob * reward
                p[valid_s_idx][valid_next_s_idx] += action_prob

        if save_path:
            state_labels = [f's{i}' for i in range(1, num_valid_states + 1)]
            df_r = pd.DataFrame(r, columns=['Reward'], index=state_labels)
            df_p = pd.DataFrame(p, index=state_labels, columns=state_labels)
            
            df_r = df_r.round(2)
            df_p = df_p.round(2)

            fig, axes = plt.subplots(2, 1, figsize=(12, 16))
            fig.tight_layout(pad=6.0)

            axes[0].axis('tight')
            axes[0].axis('off')
            table_r = axes[0].table(cellText=df_r.values, colLabels=df_r.columns,
                                rowLabels=df_r.index, cellLoc='center', loc='center')
            table_r.auto_set_font_size(False)
            table_r.set_fontsize(10)
            table_r.scale(1, 1.5)
            axes[0].set_title("Reward Vector (r_pi)", fontsize=16)

            axes[1].axis('tight')
            axes[1].axis('off')
            table_p = axes[1].table(cellText=df_p.values, colLabels=df_p.columns,
                                rowLabels=df_p.index, cellLoc='center', loc='center')
            table_p.auto_set_font_size(False)
            table_p.set_fontsize(10)
            table_p.scale(1, 1.5)
            axes[1].set_title("State Transition Matrix (P_pi)", fontsize=16)

            plt.savefig(save_path, bbox_inches='tight')
            print(f"r and P values table saved to {save_path}")
            plt.close(fig)

        return r, p


    def add_policy(self, policy_matrix):
        for state, state_action_group in enumerate(policy_matrix):
            x = state % self.env_size[0]
            y = state // self.env_size[0]
            for i, action_probability in enumerate(state_action_group):
                if action_probability !=0:
                    dx, dy = self.action_space[i]
                    if (dx, dy) != (0,0):
                        self.ax.add_patch(patches.FancyArrow(x, y, dx=(0.1+action_probability/2)*dx, dy=(0.1+action_probability/2)*dy, color=self.color_policy, width=0.001, head_width=0.05))
                    else:
                        self.ax.add_patch(patches.Circle((x, y), radius=0.07, facecolor=self.color_policy, edgecolor=self.color_policy, linewidth=1, fill=False))


    def add_state_values(self, policy_matrix, gamma=0.9, solution="closed", precision=2, save_path=None):
        r, p = self.get_r_and_p(policy_matrix)
        num_valid_states = p.shape[0]
        values = np.zeros(num_valid_states)

        if solution == "closed":
            I = np.identity(num_valid_states)
            try:
                values = np.linalg.inv(I - gamma * p) @ r
            except np.linalg.LinAlgError:
                print("Error: The matrix (I - gamma*P) is singular and cannot be inverted.")
                return

        elif solution == "iterative":
            max_iterations = 1000
            tolerance = 1e-6
            
            for i in range(max_iterations):
                v_new = r + gamma * (p @ values)
                if np.max(np.abs(v_new - values)) < tolerance:
                    print(f"Iterative solution converged in {i+1} steps.")
                    values = v_new
                    break
                
                values = v_new
            else:
                print("Iterative solution did not converge within max iterations.")

        forbidden_grid_indices = {x + y * self.env_size[0] for x, y in self.forbidden_states}
        valid_to_grid_map = [idx for idx in range(self.num_states) if idx not in forbidden_grid_indices]

        for valid_s_idx, value in enumerate(values):
            grid_s_idx = valid_to_grid_map[valid_s_idx]
            x = grid_s_idx % self.env_size[0]
            y = grid_s_idx // self.env_size[0]
            
            formatted_value = f"{value:.{precision}f}"
            self.ax.text(x, y, formatted_value, ha='center', va='center', fontsize=10, color='black')

        self.render(animation_interval=0.1) 
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            print(f"Image with state values saved to {save_path}")

        return values