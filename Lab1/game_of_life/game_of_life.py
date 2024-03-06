import numpy as np
from numba import jit
import matplotlib.pyplot as plt
import matplotlib.animation as animation
plt.rcParams['animation.ffmpeg_path'] = r'/usr/bin/ffmpeg'

class GameOfLife:
    def __init__(self, initial_state):
        self.state = initial_state

    @staticmethod
    @jit(nopython=True)
    def step(state):
        new_state = state.copy()
        for i in range(1, state.shape[0] - 1):
            for j in range(1, state.shape[1] - 1):
                total = np.sum(state[i-1:i+2, j-1:j+2]) - state[i, j]
                if state[i, j]:
                    if total < 2 or total > 3:
                        new_state[i, j] = 0
                elif total == 3:
                    new_state[i, j] = 1
        return new_state

    def run(self, steps):
        for _ in range(steps):
            self.state = self.step(self.state)

    def animate(self, steps, filename='game_of_life.mp4'):
        fig = plt.figure()
        ims = []
        WriterClass = animation.writers['ffmpeg']
        writer = WriterClass(fps=10, metadata=dict(artist='bww'), bitrate=1800)
        for _ in range(steps):
            ims.append((plt.imshow(self.state, cmap='Purples'),))
            self.state = self.step(self.state)
        ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
        ani.save(filename, writer=writer)
        

# Create a simulation with a random initial state
initial_state = np.random.randint(0, 2, (256, 512))
game = GameOfLife(initial_state)

# Run the simulation for 100 steps
game.run(100)

# Animate the simulation
game.animate(100)