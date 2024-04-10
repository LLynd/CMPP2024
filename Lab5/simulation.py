import os, copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from numba import jit
from datetime import datetime
from tqdm import tqdm

from automata import CellularAutomaton

plt.rcParams['animation.ffmpeg_path'] = r'/usr/bin/ffmpeg'


class EvolutionSimulation():
    def __init__(self, 
                 x=100,
                 y=100,):
        self.x = x
        self.y = y
        
        self.main_grid = np.random.randint(0, 2, 10, dtype=bool)
        self.main_grid.astype('u1')