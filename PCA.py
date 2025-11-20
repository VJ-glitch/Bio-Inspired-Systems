import numpy as np
import matplotlib.pyplot as plt

# ----- Step 1: Define problem -----
def sphere_function(position):
    x, y = position
    return x**2 + y**2

# ----- Step 2: Initialize parameters -----
grid_size = (10, 10)       # 10x10 grid
iterations = 100
search_space = (-10, 10)   # Value range

# ----- Step 3: Initialize population -----
grid = np.random.uniform(search_space[0], search_space[1], (grid_size[0], grid_size[1], 2))

# ----- Step 4: Evaluate fitness -----
def evaluate_fitness(grid):
    fitness = np.zeros((grid_size[0], grid_size[1]))
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            fitness[i, j] = sphere_function(grid[i, j])
    return fitness

# ----- Step 5: Neighborhood update rule -----
def get_neighbors(i, j):
    # Moore neighborhood (8 neighbors)
    neighbors = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue
            ni, nj = (i + dx) % grid_size[0], (j + dy) % grid_size[1]
            neighbors.append((ni, nj))
    return neighbors

def update_cells(grid, fitness):
    new_grid = np.copy(grid)
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            neighbors = get_neighbors(i, j)
            best_neighbor = min(neighbors, key=lambda n: fitness[n])
            new_grid[i, j] = (grid[i, j] + grid[best_neighbor]) / 2  # average update
    return new_grid

# ----- Step 6: Iteration -----
best_values = []
for _ in range(iterations):
    fitness = evaluate_fitness(grid)
    best_values.append(np.min(fitness))
    grid = update_cells(grid, fitness)

# ----- Step 7: Output result -----
final_fitness = evaluate_fitness(grid)
best_solution = grid[np.unravel_index(np.argmin(final_fitness), grid_size)]
print("Best Solution Found:", best_solution)
print("Minimum Value:", np.min(final_fitness))

# ----- Visualization -----
plt.plot(best_values)
plt.title("Convergence Curve of Parallel Cellular Algorithm")
plt.xlabel("Iteration")
plt.ylabel("Best Fitness Value")
plt.show()
