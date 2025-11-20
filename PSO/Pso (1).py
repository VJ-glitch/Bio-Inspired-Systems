import random

def objective(charge_discharge_schedule):
    cost = 0

    storage_capacity = 100
    current_storage = 0
    penalty = 0

    for action in charge_discharge_schedule:
        current_storage += action

        if current_storage > storage_capacity:
            penalty += (current_storage - storage_capacity) * 10
            current_storage = storage_capacity
        elif current_storage < 0:
            penalty += abs(current_storage) * 10
            current_storage = 0

        cost += abs(action) * 0.1

    return cost + penalty

num_particles = 20
num_iterations = 100
inertia = 0.7
personal_weight = 1.5
social_weight = 1.5
schedule_length = 24

particle_positions = [[random.uniform(-20, 20) for _ in range(schedule_length)] for _ in range(num_particles)]
particle_velocities = [[random.uniform(-5, 5) for _ in range(schedule_length)] for _ in range(num_particles)]
personal_best_positions = [list(pos) for pos in particle_positions]
personal_best_scores = [objective(pos) for pos in particle_positions]

best_score = min(personal_best_scores)
best_index = personal_best_scores.index(best_score)
global_best_position = list(personal_best_positions[best_index])

for iteration in range(num_iterations):
    for i in range(num_particles):
        for dim in range(schedule_length):
            r1 = random.random()
            r2 = random.random()
            personal_component = personal_weight * r1 * (personal_best_positions[i][dim] - particle_positions[i][dim])
            social_component = social_weight * r2 * (global_best_position[dim] - particle_positions[i][dim])
            particle_velocities[i][dim] = inertia * particle_velocities[i][dim] + personal_component + social_component

        particle_positions[i] = [particle_positions[i][dim] + particle_velocities[i][dim] for dim in range(schedule_length)]

        score = objective(particle_positions[i])

        if score < personal_best_scores[i]:
            personal_best_positions[i] = list(particle_positions[i])
            personal_best_scores[i] = score
            if score < objective(global_best_position):
                global_best_position = list(particle_positions[i])

    if iteration % 10 == 0:
        print("Iteration:", iteration, "Best Score:", objective(global_best_position))


print("Final Best Score:", objective(global_best_position))
