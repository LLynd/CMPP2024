import os, copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from numba import jit
from datetime import datetime
from tqdm import tqdm

from numba_functions import (apply_boundary_conditions, 
                             calculate_density_distribution, 
                             calculate_velocity_distribution, 
                             calculate_equilibrium_distribution, 
                             calculate_collisions, 
                             apply_collisions,
                             streaming)

plt.rcParams['animation.ffmpeg_path'] = r'/usr/bin/ffmpeg'


class LBMSimulation():
    def __init__(self, 
                 steps=20000, 
                 grid_x=520,#180 
                 grid_y=180,#520 
                 velocity=.04, 
                 Re=220, 
                 viscosity=None, 
                 relaxation_time=None, 
                 velocity_perturbation=.0001, 
                 obstacle_type='wedge',
                 n_frame_save=100,
                 dtype=np.float32):
        self.steps = steps
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.velocity = velocity
        self.Re = Re
        
        if viscosity is None:
            self.viscosity = self.velocity * self.grid_y * 0.5 / self.Re
        if relaxation_time is None:
            self.relaxation_time = 3 * self.viscosity + 0.5
        
        self.velocity_perturbation = velocity_perturbation
        self.obstacle_type = obstacle_type
        self.n_frame_save = n_frame_save
        self.dtype = dtype
        
        self.date = datetime.now().strftime("%Y%m%d%H%M%S")
        
        self.init_grids()
        self.init_obstacles(self.obstacle_type)
    
     
    def init_grids(self):
        self.directions = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]])
        self.directions = np.reshape(self.directions, (9, 2))
        
        self.reverse_directions = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])
        
        self.density_grid = np.ones((self.grid_x, self.grid_y), dtype=self.dtype)
        
        self.velocity_grid = np.zeros((self.grid_x, self.grid_y, 2), dtype=self.dtype)
        self.velocity_grid[:, :, 0] = self.velocity * (1 + self.velocity_perturbation * np.sin(2 * np.pi * np.arange(self.grid_y) / self.grid_y))
        self.velocity_grid[:, :, 1] = 0
        self.init_velocity_grid = copy.copy(self.velocity_grid)
        
        self.W = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])
        
        self.density_population_grid = np.ones((9, self.grid_x, self.grid_y), dtype=self.dtype)
        for i in range(9):
            coeff1 = self.W[i] * self.density_grid
            coeff2 = 3 * np.dot(self.velocity_grid, self.directions[i, :])
            coeff3 = 9/2 * np.square(np.dot(self.velocity_grid, self.directions[i, :]))
            coeff4 = 3/2 * (np.square(self.velocity_grid[:, :, 0]) + np.square(self.velocity_grid[:, :, 1]))
            self.density_population_grid[i, :, :] = coeff1 * (1 + coeff2 + coeff3 - coeff4)
        self.density_population_grid_eq = copy.copy(self.density_population_grid)
        self.density_population_grid_col = copy.copy(self.density_population_grid)
        
        self.grid_history = np.zeros((self.steps+1, self.grid_x, self.grid_y), dtype=self.dtype)
        abs_velocity = np.abs(self.velocity_grid)
        his_velocity = np.square(abs_velocity[:, :, 0]) + np.square(abs_velocity[:, :, 1])
        self.grid_history[0] = copy.copy(his_velocity)
    
     
    def init_obstacles(self, obstacle=None):
        if obstacle is None:
            return None #TODO: add safeguards against this later in the code
        
        elif obstacle == 'wedge':
            def fun(x, y):
                center_x = self.grid_x // 4
                center_y = self.grid_y // 2
                return np.abs(x - center_x) + np.abs(y) < center_y
            size = (self.grid_x, self.grid_y)
            wedge = np.fromfunction(fun, size, dtype=float)
            self.obstacle = wedge
            
        elif obstacle == 'cylinder':
            def fun(x, y):
                center_x = self.grid_x // 2
                center_y = self.grid_y // 2
                radius = min(self.grid_x, self.grid_y) // 4  # Adjust the radius here
                return np.square(x - center_x) + np.square(y - center_y) < np.square(radius)
            size = (self.grid_x, self.grid_y)
            cylinder = np.fromfunction(fun, size, dtype=float)
            self.obstacle = cylinder

        self.obstacle[:, 0] = True
        self.obstacle[:, -1] = True 
    
    def calculate_inlet_density_distribution(self):
        s1 = self.density_population_grid[3, :, :] + self.density_population_grid[6, :, :] + self.density_population_grid[7, :, :]
        s2 = self.density_population_grid[0, :, :] + self.density_population_grid[2, :, :] + self.density_population_grid[4, :, :]
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
        
    
    def apply_boundary_conditions(self):
        for i in [1, 5, 8]:
            self.density_population_grid[i, 0, :] = self.density_population_grid_eq[i, 0, :]
        for i in [3, 6, 7]:
            self.density_population_grid[i, self.grid_x-1, :] = self.density_population_grid[i, self.grid_x-2, :]
    
    
    def calculate_density_distribution(self):
        self.density_grid = np.sum(self.density_population_grid, axis=0)
    
    
    def calculate_velocity_distribution(self):
        coeff = 1 / self.density_grid
        #for i in range(9):
        #    temp[:,:,0] += self.directions[i, 0] * self.density_population_grid[i, :, :]
        #    temp[:,:,1] += self.directions[i, 1] * self.density_population_grid[i, :, :]
        self.velocity_grid = coeff[:, :, None] * np.tensordot(self.density_population_grid, self.directions, axes=([0],[0]))
    
    
    def calculate_equilibrium_distribution(self):
        for i in range(9):
            coeff1 = self.W[i] * self.density_grid
            coeff2 = 3 * np.dot(self.velocity_grid, self.directions[i, :])
            coeff3 = 9/2 * np.square(np.dot(self.velocity_grid, self.directions[i, :]))
            coeff4 = 3/2 * (np.square(self.velocity_grid[:, :, 0]) + np.square(self.velocity_grid[:, :, 1]))
            self.density_population_grid_eq[i, :, :] = coeff1 * (1 + coeff2 + coeff3 - coeff4)
         
            
    def calculate_collisions(self):
        for i in range(9):
            self.density_population_grid_col[i, :, :] = self.density_population_grid[i, :, :] - (self.density_population_grid[i, :, :] - self.density_population_grid_eq[i, :, :]) / self.relaxation_time
    
    
    def apply_collisions(self):
        for i in range(9):
            self.density_population_grid[i, :, :] = np.where(
                self.obstacle,
                self.density_population_grid[self.reverse_directions[i], :, :],
                self.density_population_grid_col[i, :, :]
            )
    
    
    def streaming(self):
        for i in range(9):
            self.density_population_grid[i, :, :] = np.roll(self.density_population_grid[i, :, :], self.directions[i, :], axis=(0, 1))
    
    
    def plot_state(self, s, i):
        abs_velocity = np.abs(self.velocity_grid)
        his_velocity = np.square(abs_velocity[:, :, 0]) + np.square(abs_velocity[:, :, 1])
        plt.imshow(his_velocity, cmap='hot')
        plt.savefig('results/step_{}_{}.png'.format(s, i))
        plt.cla()
        
    def step(self, s):
        #i = 0
        self.calculate_inlet_density_distribution()
        #self.plot_state(s, i)
        self.calculate_inlet_equilibrium_distribution()
        #self.plot_state(s, i+1)
        self.apply_boundary_conditions()
        self.calculate_density_distribution()
        self.calculate_velocity_distribution()
        self.calculate_equilibrium_distribution()
        self.calculate_collisions()
        self.apply_collisions()
        self.streaming()
    
        abs_velocity = np.abs(self.velocity_grid)
        his_velocity = np.square(abs_velocity[:, :, 0]) + np.square(abs_velocity[:, :, 1])
        self.grid_history[s] = copy.copy(his_velocity)
    
    
    def gpu_step(self, s):
        self.calculate_inlet_density_distribution()
        self.calculate_inlet_equilibrium_distribution()#numbyfi it
        self.density_population_grid = apply_boundary_conditions(self.density_population_grid, 
                                                                 self.density_population_grid_eq, 
                                                                 self.grid_x)
        self.density_grid = calculate_density_distribution(self.density_grid, 
                                                           self.density_population_grid)
        self.velocity_grid = calculate_velocity_distribution(self.density_grid, 
                                                             self.density_population_grid, 
                                                             self.velocity_grid, 
                                                             self.directions)
        self.density_population_grid_eq = calculate_equilibrium_distribution(self.W, 
                                                                             self.density_grid, 
                                                                             self.velocity_grid, 
                                                                             self.directions, 
                                                                             self.density_population_grid_eq)
        self.density_population_grid_col = calculate_collisions(self.density_population_grid, 
                                                            self.density_population_grid_col, 
                                                            self.density_population_grid_eq, 
                                                            self.relaxation_time)
        self.density_population_grid = apply_collisions(self.density_population_grid, 
                                                        self.density_population_grid_col, 
                                                        self.obstacle, 
                                                        self.reverse_directions)
        self.density_population_grid = streaming(self.density_population_grid, 
                                                 self.directions)
    
        abs_velocity = np.abs(self.velocity_grid)
        his_velocity = np.square(abs_velocity[:, :, 0]) + np.square(abs_velocity[:, :, 1])
        self.grid_history[s] = copy.copy(his_velocity)
        
        
    def save_animation(self):        
        output_file="lbm_flow_{}_s{}_re{}_v{}_d{}.mp4".format(self.obstacle_type, 
                                                      self.steps, 
                                                      self.Re,
                                                      self.velocity,
                                                      self.date)
        file_path = os.path.join('results', output_file)
        fig = plt.figure()
        
        WriterClass = animation.writers['ffmpeg']
        writer = WriterClass(fps=24, metadata=dict(artist='bww'), bitrate=1800)
        
        min_his = np.min(self.grid_history)
        max_his = np.max(self.grid_history)
        #self.grid_history = (self.grid_history - min_his) / (max_his - min_his)
        #min_his, max_his = np.min(self.grid_history), np.max(self.grid_history)
               
        ims = []
        print('Creating frames...')
        for _ in tqdm(range(self.steps+1)):
            if _ % self.n_frame_save == 0: # Save every 100th frame
                self.grid_history[_][self.obstacle] = max_his #improving visibility of an obstacle
                im = plt.imshow(self.grid_history[_].T, 
                            cmap='hot',
                            vmin=min_his,
                            vmax=max_his,)
                            #animated = True)
                if _ == 0:
                    plt.xlabel('X')
                    plt.ylabel('Y')
                    plt.xticks([])
                    plt.yticks([])
                    T = plt.title('LBM Flow for Re={}, Initial Velocity={}'.format(self.Re, self.velocity))

                ims.append([im])
        
        print('Creating animation...')
        ani = animation.ArtistAnimation(fig, ims, interval=10, blit=True, repeat_delay=1000)
        print('Saving animation...')
        ani.save(file_path, writer=writer)
    
    
    def save_init_state(self):
        output_file="lbm_init_{}_s{}_re{}_v{}_d{}.png".format(self.obstacle_type, 
                                                      self.steps, 
                                                      self.Re,
                                                      self.velocity,
                                                      self.date)
        file_path = os.path.join('results', output_file)
        plt.imshow(self.grid_history[0].T, cmap='hot')
        plt.savefig(file_path)
        plt.cla()
    
    
    def save_state(self, s):
        plt.imshow(self.grid_history[s], cmap='hot')
        plt.savefig('results/step_{}.png'.format(s))
        plt.cla()
                
                
    def run(self, gpu=False, save_frames=False):
        print('Running LBM simulation...')
        self.save_init_state()
        if gpu is False:
            for s in tqdm(range(1, self.steps+1)):
                self.step(s)
                if save_frames:
                        if s % self.n_frame_save == 0:
                            self.save_state(s)
        else:
            for s in tqdm(range(1, self.steps+1)):
                self.gpu_step(s)
                if save_frames:
                        if s % self.n_frame_save == 0:
                            self.save_state(s)
        self.save_init_state()
        self.save_animation()
        print('Simulation finished.')
            