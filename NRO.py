import time
import numpy as np

def NRO(population,sphere_function,lower_limit,upper_limit,max_iterations):
    population_size,num_variables = population.shape[0],population.shape[1]
    alpha = 0.1
    beta = 0.5
    gamma = 0.1
    conv = np.zeros((max_iterations))
    ct = time.time()
    for iteration in range(max_iterations):
        # Evaluate objective function for each individual
        fitness = np.array([sphere_function(ind) for ind in population])

        # Sort population based on fitness
        sorted_indices = np.argsort(fitness)
        population = population[sorted_indices]
        fitness = fitness[sorted_indices]

        # Update each individual's position
        for i in range(population_size):
            for j in range(num_variables):
                for k in range(population_size):
                    if k != i:
                        r = np.linalg.norm(population[k] - population[i])
                        if r != 0:
                            direction = (population[k, j] - population[i, j]) / r
                            population[i, j] += alpha * beta * direction + gamma * np.random.rand() * (
                                        upper_limit - lower_limit)

            # Ensure the new position is within bounds
            population[i] = np.clip(population[i], lower_limit, upper_limit)
        # Find the best solution
        best_solution = population[0]
        best_fitness = np.min(fitness)
        conv[iteration] = best_fitness
    ct = time.time()-ct
    return best_fitness,conv,best_solution,ct
