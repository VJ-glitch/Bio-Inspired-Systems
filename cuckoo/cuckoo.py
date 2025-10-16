import numpy as np
import random

def total_distance(route, distance_matrix):
    dist = 0
    for i in range(len(route)):
        dist += distance_matrix[route[i-1]][route[i]]
    return dist

def random_swap(route):
    new_route = route.copy()
    i, j = np.random.choice(len(route), 2, replace=False)
    new_route[i], new_route[j] = new_route[j], new_route[i]
    return new_route

def cuckoo_search_tsp(distance_matrix, n_nests=15, pa=0.25, max_iter=300):
    n_cities = len(distance_matrix)
    nests = [random.sample(range(n_cities), n_cities) for _ in range(n_nests)]
    fitness = [total_distance(route, distance_matrix) for route in nests]
    best_idx = np.argmin(fitness)
    best_route = nests[best_idx]
    best_fitness = fitness[best_idx]
    for t in range(max_iter):
        for i in range(n_nests):
            new_route = random_swap(nests[i])
            new_fitness = total_distance(new_route, distance_matrix)
            if new_fitness < fitness[i]:
                nests[i] = new_route
                fitness[i] = new_fitness
        n_abandon = int(pa * n_nests)
        worst_idx = np.argsort(fitness)[-n_abandon:]
        for i in worst_idx:
            nests[i] = random.sample(range(n_cities), n_cities)
            fitness[i] = total_distance(nests[i], distance_matrix)
        current_best_idx = np.argmin(fitness)
        if fitness[current_best_idx] < best_fitness:
            best_route = nests[current_best_idx]
            best_fitness = fitness[current_best_idx]
    return best_route, best_fitness

n = int(input("Enter number of cities: "))
print("Enter distance matrix:")
distance_matrix = []
for i in range(n):
    row = list(map(int, input().split()))
    distance_matrix.append(row)
distance_matrix = np.array(distance_matrix)
route, dist = cuckoo_search_tsp(distance_matrix)
print("Best Route:", route)
print("Shortest Distance:", dist)