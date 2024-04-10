import numpy as np
from numba import jit


@jit(nopython=True)
def apply_boundary_conditions(density_population_grid, 
                              density_population_grid_eq, 
                              grid_x):
    for i in [1, 5, 8]:
        density_population_grid[i, 0, :] = density_population_grid_eq[i, 0, :]
    for i in [3, 6, 7]:
        density_population_grid[i, grid_x-1, :] = density_population_grid[i, grid_x-2, :]
    return density_population_grid


@jit(nopython=True)
def calculate_density_distribution(density_grid, density_population_grid):
    density_grid = np.sum(density_population_grid, axis=0)
    return density_grid


@jit(nopython=True)
def calculate_velocity_distribution(density_grid, 
                                    density_population_grid, 
                                    velocity_grid, 
                                    directions):
    coeff = 1 / density_grid
    velocity_grid = coeff[:, :, None] * np.tensordot(density_population_grid, directions, axes=([0],[0]))
    return velocity_grid


@jit(nopython=True)
def calculate_equilibrium_distribution(W, 
                                       density_grid, 
                                       velocity_grid, 
                                       directions, 
                                       density_population_grid_eq):
    for i in range(9):
        coeff1 = W[i] * density_grid
        coeff2 = 3 * np.dot(velocity_grid, directions[i, :])
        coeff3 = 9/2 * np.square(np.dot(velocity_grid, directions[i, :]))
        coeff4 = 3/2 * (np.square(velocity_grid[:, :, 0]) + np.square(velocity_grid[:, :, 1]))
        density_population_grid_eq[i, :, :] = coeff1 * (1 + coeff2 + coeff3 - coeff4)
    return density_population_grid_eq
        
        
@jit(nopython=True)
def calculate_collisions(density_population_grid, 
                         density_population_grid_col, 
                         density_population_grid_eq, 
                         relaxation_time):
    for i in range(9):
        temp = (density_population_grid[i, :, :] - density_population_grid_eq[i, :, :]) / relaxation_time
        density_population_grid_col[i, :, :] = density_population_grid[i, :, :] - temp
    return density_population_grid_col


@jit(nopython=True)
def apply_collisions(density_population_grid, 
                     density_population_grid_col, 
                     obstacle, 
                     reverse_directions):
    for i in range(9):
        density_population_grid[i, :, :] = np.where(
            obstacle,
            density_population_grid[reverse_directions[i], :, :],
            density_population_grid_col[i, :, :]
        )
    return density_population_grid


@jit(nopython=True)
def streaming(density_population_grid, directions):
    for i in range(9):
        density_population_grid[i, :, :] = np.roll(density_population_grid[i, :, :], 
                                                   directions[i, :], axis=(0, 1))
