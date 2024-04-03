import os
import numpy as np
from random import randint
import copy
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap
from matplotlib.cm import ScalarMappable
plt.rcParams['animation.ffmpeg_path'] = r'/usr/bin/ffmpeg'

from ant import Ant


class Simulation():
    def __init__(self, num_ants, grid_width, grid_height, steps, boundary='periodic'):
        self.num_ants = num_ants
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.steps = steps
        self.boundary = boundary
        
        self.move_grid = np.zeros((grid_width, grid_height), dtype=np.dtype('u1'))
        self.pheromone_grid = np.zeros((grid_width, grid_height), dtype=np.dtype('complex'))#, dtype=np.dtype('u1'))
        self.move_history = np.zeros((steps, grid_width, grid_height), dtype=np.dtype('u1'))
        self.pheromone_history = np.zeros((steps, grid_width, grid_height), dtype=np.dtype('complex'))#, dtype=np.dtype('1'))
        self.stats_history = np.zeros((steps, 2), dtype=np.dtype('u1'))
        
        self.date = datetime.now().strftime("%Y%m%d%H%M%S")
        self.init_terrain()
        #self.init_terrain_grid = copy.copy(self.move_grid) * (1 + 1j)
        self.init_ants()
        
    def init_ants(self):
        self.ants = []
        dw = self.grid_width//10
        dh = self.grid_height//10
        for _ in range(self.num_ants):
            i = 0
            sp = (randint(dw*2, dw*5), randint(dh*2, dh*3))
            while self.move_grid[sp[0], sp[1]] != 0 and i < 1000:
                sp = (randint(dw*2, dw*5), randint(dh*2, dh*3))
                i += 1
            if i == 1000: #Apply movinng ants to free space
                self.init_move_ants()
                self.init_move_ants()
            self.ants.append(Ant(self.grid_width, self.grid_height, sp))
            self.move_grid[sp[0], sp[1]] = 64#1
        
    def init_terrain(self):
        dw = self.grid_width//10
        dh = self.grid_height//10
        
        nest_x1 = dw
        nest_x2 = dw*2
        nest_y1 = dh
        nest_y2 = dh*2
        
        food_x1 = dw * 8
        food_x2 = dw * 9
        food_y1 = dh * 8
        food_y2 = dh * 9
        
        self.move_grid[nest_x1:nest_x2, nest_y1:nest_y2] = 128
        self.move_grid[food_x1:food_x2, food_y1:food_y2] = 255
        #self.move_grid[dw*5:dw*6, dh*3:dh*8] = 64
        #self.move_grid[dw*4:dw*9, dh*4:dh*5] = 64
        #self.move_grid[dw*2:dw*7, dh*6:dh*7] = 64
        
        if self.boundary == 'periodic':
            self.move_grid[0, :] = 64
            self.move_grid[-1, :] = 64
            self.move_grid[:, 0] = 64
            self.move_grid[:, -1] = 64
    
    def init_move_ants(self):
        for ant in self.ants:
            cells_with_contents = [(c, self.move_grid[c], self.pheromone_grid[c]) for c in ant.get_cells_in_front()]
            chosen_cell = ant.choose_destination(cells_with_contents)
            p = ant.get_position()
            pr = ant.get_previous_position()
            ant.set_position(chosen_cell[0], chosen_cell[1])
            self.move_grid[pr[0], pr[1]] = 0
            self.move_grid[chosen_cell[0], chosen_cell[1]] = 64
        
    def get_ants_status(self):
        return [ant.get_status() for ant in self.ants]
    
    def evaporate_pheromones(self):
        self.pheromone_grid = self.pheromone_grid * 0.99
        
    def move_ant(self, ant, step):
        cells_with_contents = [(c, self.move_grid[c], self.pheromone_grid[c]) for c in ant.get_cells_in_front()]
        chosen_cell = ant.choose_destination(cells_with_contents)
        ant.emit_pheromone()
        p = ant.get_position()
        pr = ant.get_previous_position()
        self.pheromone_grid[p[0], p[1]] += ant.get_last_pheromone()
        ant.set_position(chosen_cell[0], chosen_cell[1])
        if step >= 1:
            self.move_grid[pr[0], pr[1]] = 0
        
        self.move_grid[chosen_cell[0], chosen_cell[1]] = 64#1
        
    def save_animation(self):
        output_file = "ants_history_n{num_ants}_s{steps}_{d}.mp4".format(num_ants=self.num_ants, steps=self.steps,
                                                                        d=self.date)
        file_path = os.path.join('results', output_file)
        cmp = ListedColormap(['grey', 'black', 'green', 'yellow'])
        fig, ax = plt.subplots(2, 2, figsize=(16, 8))

        WriterClass = animation.writers['ffmpeg']
        writer = WriterClass(fps=8, metadata=dict(artist='bww'), bitrate=1800)
        
        ims = []

        # Find the overall min and max values for each dataset
        move_min = np.min(self.move_history)
        move_max = np.max(self.move_history)
        pheromone_min = np.min(np.real(self.pheromone_history))
        pheromone_max = np.max(np.real(self.pheromone_history))
        pheromone_imag_min = np.min(np.imag(self.pheromone_history))
        pheromone_imag_max = np.max(np.imag(self.pheromone_history))

        # Initialize legends
        ax[0, 1].plot([], [], label='Foraging')
        ax[0, 1].plot([], [], label='Returning')
        L = ax[0, 1].legend()

        def update(frame):
            ims.clear()
            ax[0, 0].cla() 
            im0 = ax[0, 0].imshow(self.move_history[frame], cmap=cmp, vmin=move_min, vmax=move_max)
            ax[0, 0].text(0.1, 0.87, f'{self.stats_history[frame, 0]}', color='Blue', transform=ax[0, 0].transAxes, fontsize=10, ha='left', va='top')
            ax[0, 0].text(0.9, 0.13, f'{self.stats_history[frame, 1]}', color='Blue', transform=ax[0, 0].transAxes, fontsize=10, ha='right', va='bottom')
            ax[0, 1].cla() 
            ax[0, 1].plot(self.stats_history[:frame, 0], label='Foraging')
            ax[0, 1].plot(self.stats_history[:frame, 1], label='Returning')
            ax[0, 1].legend()
            im2 = ax[1, 0].imshow(np.real(self.pheromone_history[frame]), vmin=pheromone_min, vmax=pheromone_max)
            im3 = ax[1, 1].imshow(np.imag(self.pheromone_history[frame]), vmin=pheromone_imag_min, vmax=pheromone_imag_max)
            ims.extend([im0, im2, im3])
            return ims

        print('Creating animation...')
        ani = animation.FuncAnimation(fig, update, frames=self.steps, interval=50, blit=True)
        print('Saving animation...')
        ani.save(file_path, writer=writer)
        
    def save_stats_plots(self):
        output_file="ants_stats_n{num_ants}_s{steps}_{d}.png".format(num_ants=self.num_ants, steps=self.steps, d=self.date)
        p = os.path.join('results', output_file)
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        cmp = ListedColormap(['grey', 'black', 'green', 'yellow'])
        ax[0].plot(self.stats_history[:, 0], label = 'Foraging')
        ax[0].plot(self.stats_history[:, 1], label = 'Returning')
        ax[1].imshow(self.move_history[-1], cmap=cmp, vmin=0, vmax=256, animated = True)
        ax[0].legend()
        plt.savefig(p)
    
    def run(self):
        print('Running simulation...')
        for i in tqdm(range(self.steps)):
            for ant in self.ants:
                self.move_ant(ant, i)
            self.evaporate_pheromones()
            self.move_history[i] = self.move_grid
            self.pheromone_history[i] = self.pheromone_grid
            self.stats_history[i, 0] = self.get_ants_status().count(True)
            self.stats_history[i, 1] = self.get_ants_status().count(False)
        self.save_stats_plots()
        #self.save_animation()
        self.save_animation_fast()
        print('Done')

    def save_animation_fast(self):
        output_file="ants_history_n{num_ants}_s{steps}_{d}.mp4".format(num_ants=self.num_ants, steps=self.steps, d=self.date)
        file_path = os.path.join('results', output_file)
        cmp = ListedColormap(['grey', 'black', 'green', 'yellow'])
        fig, ax = plt.subplots(1, 3, figsize=(12, 6))
        ims = []
        WriterClass = animation.writers['ffmpeg']
        writer = WriterClass(fps=8, metadata=dict(artist='bww'), bitrate=1800)
        for _ in tqdm(range(self.steps)):
            im0 = ax[0].imshow(self.move_history[_], cmap=cmp, vmin=0, vmax=256, animated = True)
            im1 = ax[1].imshow(np.real(self.pheromone_history[_]), animated = True)
            im2 = ax[2].imshow(np.imag(self.pheromone_history[_]), animated = True)
            ims.append([im0, im1, im2])#, im2])
        print('Saving animation...')
        ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
        ani.save(file_path, writer=writer) 
    
    def get_simulation_stats(self):
        pass
    
    def plot_simulation_stats(self):
        pass
    