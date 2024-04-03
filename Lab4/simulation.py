import os, copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from numba import jit
from datetime import datetime
from tqdm import tqdm
from matplotlib.colors import ListedColormap

plt.rcParams['animation.ffmpeg_path'] = r'/usr/bin/ffmpeg'


class LBMSimulation():
    def __init__(self, 
                 steps=20000, 
                 grid_x=180, 
                 grid_y=520, 
                 velocity=.04, 
                 Re=220, 
                 viscosity=None, 
                 relaxation_time=None, 
                 velocity_perturbation=.0001, 
                 obstacle_type='wedge'):
        self.steps = steps
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.velocity = velocity
        self.Re = Re
        
        if viscosity is None:
            self.viscosity = velocity * grid_x * 0.5 / Re
        if relaxation_time is None:
            self.relaxation_time = 3 * self.viscosity + 0.5
        
        self.velocity_perturbation = velocity_perturbation
        self.obstacle_type = obstacle_type
        
        self.date = datetime.now().strftime("%Y%m%d%H%M%S")
        
        self.init_grids()
        self.init_obstacles(self.obstacle_type)
    
     
    def init_grids(self):
        self.directions = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]])
        self.directions = np.reshape(self.directions, (9, 2))
        
        self.reverse_directions = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])
        
        self.density_grid = np.ones((self.grid_x, self.grid_y))
        
        self.velocity_grid = np.zeros((self.grid_x, self.grid_y, 2))
        self.velocity_grid[:, :, 0] = self.velocity * (1 + np.dot(self.velocity_perturbation, np.sin(2 * np.pi * np.arange(self.grid_y) / self.grid_y)))
        self.velocity_grid[:, :, 1] = 0
        self.init_velocity_grid = copy.copy(self.velocity_grid)
        
        self.W = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])
        
        self.density_population_grid = np.ones((9, self.grid_x, self.grid_y))
        for i in range(9):
            coeff1 = self.W[i] * self.density_grid
            coeff2 = 3 * np.dot(self.velocity_grid, self.directions[i, :])
            coeff3 = 9/2 * np.square(np.dot(self.velocity_grid, self.directions[i, :]))
            coeff4 = 3/2 * (np.square(self.velocity_grid[:, :, 0]) + np.square(self.velocity_grid[:, :, 1]))
            self.density_population_grid[i, :, :] = coeff1 * (1 + coeff2 + coeff3 - coeff4)
        self.density_population_grid_eq = copy.copy(self.density_population_grid)
        
        self.grid_history = np.zeros((self.steps, self.grid_x, self.grid_y))
    
     
    def init_obstacles(self, obstacle=None):
        if obstacle is None:
            return None
        
        elif obstacle == 'wedge':
            def fun(x, y):
                center_x = self.grid_x // 2
                center_y = self.grid_y // 2
                return (y > center_y - x + center_x) & (y < center_y + x - center_x)
            size = (self.grid_x, self.grid_y)
            wedge = np.fromfunction(fun, size, dtype=float)
            self.obstacle = wedge
            
        elif obstacle == 'cylinder':
            def fun(x, y):
                center_x = self.grid_x // 2
                center_y = self.grid_y // 2
                radius = min(self.grid_x, self.grid_y) // 4  # Adjust the radius as needed
                return np.square(x - center_x) + np.square(y - center_y) < np.square(radius)
            size = (self.grid_x, self.grid_y)
            cylinder = np.fromfunction(fun, size, dtype=float)
            self.obstacle = cylinder

    
    def calculate_inlet_density_distribution(self):
        tiny = np.finfo(float).tiny
        large = np.finfo(float).max
        self.density_population_grid = np.clip(self.density_population_grid, tiny, large)

        s1 = self.density_population_grid[3, :, :] + self.density_population_grid[6, :, :] + self.density_population_grid[7, :, :]
        s2 = self.density_population_grid[1, :, :] + self.density_population_grid[5, :, :] + self.density_population_grid[8, :, :]
        num = 2 * s1 + s2
        denom = 1 - np.abs(self.init_velocity_grid[0, :, 1])
        div = np.divide(num, denom)
        self.density_grid[0, :] = div[0, :]
            
    
    def calculate_inlet_equilibrium_distribution(self):
        for i in range(9):
            coeff1 = self.W[i] * self.density_grid[0, :]
            coeff2 = 3 * np.dot(self.init_velocity_grid, self.directions[i, :])
            coeff3 = 9/2 * np.square(np.dot(self.init_velocity_grid, self.directions[i, :]))
            coeff4 = 3/2 * (np.square(self.init_velocity_grid[:, :, 0]) + np.square(self.init_velocity_grid[:, :, 1]))
            res = coeff1 * (1 + coeff2 + coeff3 - coeff4)
            self.density_population_grid_eq[i, 0, :] = res[0, :]
        
    
    #@jit(nopython=True)
    def apply_boundary_conditions(self):
        for i in [1, 5, 8]:
            self.density_population_grid[i, 0, :] = self.density_population_grid_eq[i, 0, :]
        for i in [3, 6, 7]:
            self.density_population_grid[i, self.grid_x-1, :] = self.density_population_grid[i, self.grid_x-2, :]
    
    
    #@jit(nopython=True)
    def calculate_density_distribution(self):
        self.density_grid = np.sum(self.density_population_grid, axis=0)
    
    
    #@jit(nopython=True)
    def calculate_velocity_distribution(self):
        coeff = 1 / self.density_grid
        self.density_population_grid = np.nan_to_num(self.density_population_grid)
        coeff = np.nan_to_num(coeff)
        self.velocity_grid = coeff[:, :, None] * np.tensordot(self.density_population_grid, self.directions, axes=([0],[0]))
    
    
    #@jit(nopython=True)
    def calculate_equilibrium_distribution(self):
        for i in range(9):
            coeff1 = self.W[i] * self.density_grid
            coeff2 = 3 * np.dot(self.velocity_grid, self.directions[i, :])
            coeff3 = 9/2 * np.square(np.dot(self.velocity_grid, self.directions[i, :]))
            coeff4 = 3/2 * (np.square(self.velocity_grid[:, :, 0]) + np.square(self.velocity_grid[:, :, 1]))
            self.density_population_grid[i, :, :] = coeff1 * (1 + coeff2 + coeff3 - coeff4)
         
            
    #@jit(nopython=True)
    def calculate_collisions(self):
        for i in range(9):
            tiny = np.finfo(float).tiny
            large = np.finfo(float).max
            self.density_population_grid = np.clip(self.density_population_grid, tiny, large)
            self.density_grid = np.clip(self.density_grid, tiny, large)
            self.density_population_grid[i, :, :] = self.density_population_grid[i, :, :] - (self.density_population_grid[i, :, :] - self.density_grid) / self.relaxation_time
    
    
    #@jit(nopython=True)
    def apply_collisions(self):
        for i in range(9):
            self.density_population_grid[i, :, :] = np.where(
                self.obstacle,
                self.density_population_grid[self.reverse_directions[i], :, :],
                self.density_population_grid[i, :, :]
            )
    
    
    def streaming(self):
        for i in range(9):
            self.density_population_grid[i, :, :] = np.roll(self.density_population_grid[i, :, :], self.directions[i, :], axis=(0, 1))
    
    
    def step(self, s):
        tiny = np.finfo(float).tiny
        large = np.finfo(float).max
        self.density_population_grid = np.clip(self.density_population_grid, tiny, large)
        
        self.calculate_inlet_density_distribution()
        self.calculate_inlet_equilibrium_distribution()
        self.calculate_density_distribution()
        self.calculate_velocity_distribution()
        self.calculate_equilibrium_distribution()
        self.calculate_collisions()
        self.apply_collisions()
        self.streaming()
        # Use distribution function after streaming as the initial one for the next iteration
        
        abs_velocity = np.abs(self.velocity_grid)
        his_velocity = np.square(abs_velocity[:, :, 0]) + np.square(abs_velocity[:, :, 1])
        self.grid_history[s] = copy.copy(his_velocity)


    def save_animation(self):        
        output_file="lbm_flow_{}_s{}_re{}_d{}.mp4".format(self.obstacle_type, 
                                                      self.steps, 
                                                      self.Re,
                                                      self.date)
        file_path = os.path.join('results', output_file)
        fig, ax = plt.subplots(figsize=(6, 6))
        
        WriterClass = animation.writers['ffmpeg']
        writer = WriterClass(fps=8, metadata=dict(artist='bww'), bitrate=1800)
        
        self.grid_history = np.nan_to_num(self.grid_history)
        min_his = np.min(self.grid_history)
        max_his = np.max(self.grid_history)
        self.grid_history = (self.grid_history - min_his) / (max_his - min_his)
        min_his, max_his = np.min(self.grid_history), np.max(self.grid_history)
        
        print(min_his, max_his)        
        ims = []
        
        for _ in tqdm(range(self.steps)):
            if _%10 == 0: # Save every 100th frame
                im = ax.imshow(self.grid_history[_], 
                            cmap='hot',
                            vmin=min_his,
                            vmax=max_his,
                            animated = True)
                if _ == 0:
                    ax.set_xlabel('X')
                    ax.set_ylabel('Y')
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_aspect('equal')
                    T = ax.set_title('LBM Flow for Re={}'.format(self.Re))

                ims.append([im])
        
        print('Creating animation...')
        ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
        print('Saving animation...')
        ani.save(file_path, writer=writer)
    
    
    def run(self):
        print('Running LBM simulation...')
        for s in tqdm(range(self.steps)):
            self.step(s)
        self.save_animation()
        print('Simulation finished.')
            