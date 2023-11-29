import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt

def initialize_grid(dim):
    grid = np.zeros(shape=dim)
    grid[:, 0] = 100
    grid[:, -1] = 50
    grid[1:-1, 1:-1] = 50
    return grid

def plot_heatmap(grid):
    sn.heatmap(grid, cmap="hot", cbar_kws={'label': 'Temperature'})
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Heat Diffusion Simulation')
    plt.show()

def heat_diffusion_simulation(initial_grid, r, num_steps=600, plot_interval=10):
    grid = initial_grid.copy()

    for t in range(num_steps):
        new_grid = np.copy(grid)
        for i in range(1, dim[0] - 1):
            for j in range(1, dim[1] - 1):
                new_grid[i, j] = (1 - 8 * r) * grid[i, j] + r * (
                    grid[i - 1, j - 1] + grid[i - 1, j] + grid[i - 1, j + 1] +
                    grid[i, j - 1] + grid[i, j + 1] + grid[i + 1, j - 1] +
                    grid[i + 1, j] + grid[i + 1, j + 1]
                )

        new_grid[0, :] = 50
        new_grid[-1, :] = 50
        grid = new_grid

        if (t % plot_interval) == 0:
            plot_heatmap(grid)

dim = (50, 50)
r = 0.1
initial_grid = initialize_grid(dim)
heat_diffusion_simulation(initial_grid, r)
