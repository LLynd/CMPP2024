import numpy as np
from PIL import Image, ImageFilter
from numba import int32, uint8
from random import randint
from numba.experimental import jitclass


# Define the LangtonsAnt class
@jitclass([('x', int32), ('y', int32), ('direction', int32), ('grid_width', int32), ('grid_height', int32)])
class LangtonsAnt:
    def __init__(self, grid_width, grid_height):
        self.x = randint(0, grid_width - 1)
        self.y = randint(0, grid_height - 1)
        self.direction = randint(0, 3)
        self.grid_width = grid_width
        self.grid_height = grid_height

    def move(self, grid):
        if grid[self.x, self.y] == 0:
            self.direction = (self.direction + 1) % 4
        else:
            self.direction = (self.direction - 1) % 4

        grid[self.x, self.y] = 1 - grid[self.x, self.y]

        if self.direction == 0:
            self.x = (self.x + 1) % self.grid_width
        elif self.direction == 1:
            self.y = (self.y + 1) % self.grid_height
        elif self.direction == 2:
            self.x = (self.x - 1) % self.grid_width
        elif self.direction == 3:
            self.y = (self.y - 1) % self.grid_height

    def get_position(self):
        return self.x, self.y

# Define the LangtonsAntSimulation class
class LangtonsAntSimulation:
    def __init__(self, grid_width, grid_height, num_ants):
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.num_ants = num_ants
        self.grid = np.zeros((grid_width, grid_height), dtype=np.uint8)
        self.ants = [LangtonsAnt(grid_width, grid_height) for _ in range(num_ants)]
        self.paths = [[] for _ in range(num_ants)]

    def simulate(self, steps):
        for _ in range(steps):
            for i, ant in enumerate(self.ants):
                ant.move(self.grid)
                self.paths[i].append(ant.get_position())

    def visualize(self, output_file="langtons_ant.png"):
        colors = np.random.randint(0, 256, size=(self.num_ants, 3), dtype=np.uint8)
        image = np.zeros((self.grid_width, self.grid_height, 4), dtype=np.uint8)  # Add an alpha channel

        for i, path in enumerate(self.paths):
            path_color = colors[i]
            positions = np.array(path)
            for j, position in enumerate(positions):
                alpha = j / len(positions)  # Compute the alpha based on the step number
                image[position[0], position[1], :3] = path_color
                image[position[0], position[1], 3] = int(alpha * 255)  # Set the alpha channel

        img = Image.fromarray(image.transpose(1, 0, 2), 'RGBA')  # Use 'RGBA' mode for the image
        img_sharpen = img.filter(ImageFilter.SHARPEN)
        img.save(output_file, dpi=(10000, 10000))
        
# Create a simulation with a 1000x1000 grid and 10 ants
simulation = LangtonsAntSimulation(200, 200, 4)

# Run the simulation for 10000 steps
simulation.simulate(100000)

# Visualize the result
simulation.visualize("langtons_ant_simulation.tiff")