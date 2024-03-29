import numpy as np
from PIL import Image, ImageFilter
from numba import int32, uint8
from random import randint, choice

#0 - empty, 255 - food, 128 - home, 64 - obstacle/ant, 
# <0 - pheromone home, 0< - pheromone food #zmienic na prawdopodobienstwo

#Orientation ^0, /^1, ->2, \v3, v4, v/5, <-6, ^\7 - work in mod8
class Ant():
    def __init__(self, grid_width, grid_height, position=None):
        self.grid_width = grid_width
        self.grid_height = grid_height
        if position == None:
            self.position = (randint(0, self.grid_width), randint(0, self.grid_height))
        else:
            self.position = position
        
        self.move_history = [None, self.position]
        self.orientation = randint(0, 7)
        self.isForaging = True
        self.lastPheromone = 0 + 0j
        self.timeSinceChange = 1
        
    def get_position(self):
        return self.position
    
    def get_orientation(self):
        return self.orientation
    
    def get_last_pheromone(self):
        return self.lastPheromone
    
    def get_status(self):
        return self.isForaging
    
    def get_previous_position(self):
        return self.move_history[0]
    
    def set_position(self, x, y):
        p = self.get_position()
        self.position = (x, y)
    
    def set_orientation(self, o):
        self.orientation = o%8
    
    def turn_around(self):
        self.move_history[0] = self.move_history[1]
        self.move_history[1] = self.get_position()
        new_orientation = self.get_orientation() + 4
        self.set_orientation(new_orientation)
    
    def emit_pheromone(self):
        self.lastPheromone = (1 + 0j) * self.timeSinceChange if self.isForaging else (0 + 1j) * self.timeSinceChange
    
    def get_cells_in_front(self):
        x0, y0 = self.get_position()
        o = self.get_orientation()
        x1 = (x0+1)%self.grid_width
        x2 = (x0-1)%self.grid_width
        y1 = (y0+1)%self.grid_height
        y2 = (y0-1)%self.grid_height
        neighboring_cells = [(x0, y1), (x1, y1), (x1, y0), (x1, y2), (x0, y2), (x2, y2), (x2, y0), (x2, y1)]
        cells_in_front = [neighboring_cells[(o-1)%8], neighboring_cells[o%8], neighboring_cells[(o+1)%8]]
        return cells_in_front
                 
    def choose_destination(self, cells_with_contents):
        #cells_with_contents: ((x, y), terrain, pheromone)
        pheromones = []
        orientation_changes = [-1, 0, 1]
        obstacles = []
        for c in range(len(cells_with_contents)):
            if cells_with_contents[c][1] == 255 and self.isForaging:
                self.isForaging = False
                self.timeSinceChange = 1
                self.turn_around()
                chosen_cell = self.get_position()
                return chosen_cell
                
            elif cells_with_contents[c][1] == 128 and self.isForaging is False:
                self.isForaging = True
                self.timeSinceChange = 1
                self.turn_around()
                chosen_cell = self.get_position()
                return chosen_cell
            
            elif cells_with_contents[c][1] == 255 and self.isForaging is False:
                obstacles.append(c)
            
            elif cells_with_contents[c][1] == 0:
                pheromones.append([np.real(cells_with_contents[c][2]), np.imag(cells_with_contents[c][2])])
            
            else:
                obstacles.append(c)

        cells_with_contents = [e for i, e in enumerate(cells_with_contents) if i not in obstacles]
        orientation_changes = [e for i, e in enumerate(orientation_changes) if i not in obstacles]
        
        if len(cells_with_contents) == 0:
            chosen_cell = self.get_position()
            self.turn_around()
            return chosen_cell
        
        pheromones = np.array(pheromones)
        
        if self.isForaging:
            if np.random.rand() < 0.4:
                idx = randint(0, len(pheromones)-1)
            else:
                idx = np.argmax(pheromones[:, 1])
        else:
            if np.random.rand() < 0.05:
                idx = randint(0, len(pheromones)-1)
            else:
                idx = np.argmax(pheromones[:, 0])#bylo zaminione
        
        chosen_cell = cells_with_contents[idx][0] 
        new_orientation = orientation_changes[idx]
        self.move_history[0] = self.move_history[1]
        self.move_history[1] = self.get_position()
        self.set_orientation(self.get_orientation()+new_orientation)
        self.timeSinceChange = self.timeSinceChange * 0.99

        return chosen_cell
