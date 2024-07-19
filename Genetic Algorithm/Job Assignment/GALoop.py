'''
This file demonstrate how to solve a common job assignment problem
Example: A given matrix represents the cost of assigning jobs to workers. 
Each row represents a worker, and each column represents the cost of assigning a job to that worker.
cost_matrix = [
    [9, 2, 7, 8], 
    [6, 4, 3, 7], 
    [5, 8, 1, 8], 
    [7, 6, 9, 4]  
]

A deepcopy is being used to prevent changes made to mutable object

Selection Method: Roulette Wheel Selection with Elitsm
Crossover Method: Partially Mapped Crossover
Mutation Method: Swap Mutation
'''
from Individual import Individual
import random
import numpy as np
from crossover import pmx_crossover
import copy
from mutation import swap_mutation
import matplotlib.pyplot as plt
import time
from datafile import cost_matrix4 as cost_matrix


if __name__ == "__main__":
    result_objective :list = []
    result_fitness :list = []
    result_bestIngen :list = []
    
    

    # Param
    POPULATION_SIZE = 10
    WORKER_SIZE, JOB_SIZE  = np.shape(cost_matrix)
    GENERATION_THRESHOLD = 100
    OBJECTIVE_THRESHOLD = 0

    generation = 1
    solution_found = False

    # 1 = 100 %, 0.5 = 50%
    crossoverRate = 0.35
    mutationRate = 0.3
    # crossoverRate = 1
    # mutationRate = 1
    Elite_percentage = 0.2

    population :list[Individual] = []

    best_solution :Individual = Individual([0 for i in range(WORKER_SIZE)])
    best_solution.objective = float('inf')
    best_solution_gen = -float('inf')

    start = time.time()
    
    # Generating an initial population
    print("Generated Population")
    for _ in range(POPULATION_SIZE):
        individual_chromosome = [i for i in range(JOB_SIZE)]
        random.shuffle(individual_chromosome)
        individual_chromosome = individual_chromosome[:WORKER_SIZE]
        print(f"{individual_chromosome}")
        population.append(Individual(individual_chromosome)) 

    while True:
        # Calculating fitness 
        print("\nCalculating Fitness")
        total_fitness = 0
        for individual in population:
            # Calculating objective function
            cost = individual.objective_function(cost_matrix)
            fitness = individual.fitness_function(cost)
            print(f"chromosome : {individual}, objective : {individual.objective}, fitness : {individual.fitness}")
            total_fitness += fitness

        print(f"\ntotal fitness {total_fitness}")

        # Preserve elite chromosome over to the next generation
        population.sort(key=lambda x: x.fitness, reverse=True)
        Elite_count = int(POPULATION_SIZE * Elite_percentage)
        Elite = copy.deepcopy(population[:Elite_count])
        new_population = Elite

        # Calculating cumulative probability
        cumulative_probability = 0
        print(f"\nCalculating cumulative probability")
        for individual in population:
            individual.probability = individual.fitness / total_fitness
            cumulative_probability += individual.probability
            individual.cumulative_probability = cumulative_probability
            print(f"chromosome : {individual}, survive chance : {individual.probability}, cumulative : {individual.cumulative_probability}")

        # Perform roulette wheel selections
        selected_parents :list[Individual] = []
        print("\nPerforming the roulette wheel selection")
        while len(selected_parents) < POPULATION_SIZE - len(new_population):
            random_number = random.uniform(0, 1)
            for individual in population:
                if random_number < individual.cumulative_probability:
                    selected_parents.append(individual)
                    print(f"selected parent : {individual}, objective : {individual.objective}")
                    break

        # Perform crossover
        print(f"\nPerforming Crossover(PMX)")
        mated :list[Individual] = pmx_crossover(selected_parents, crossoverRate)
        for individual in mated:
            print(f'mated chromosome : {individual.chromosome}')
            
        print(f"\nPerforming swap random mutation")
        mutated :list[Individual] = swap_mutation(mated, mutationRate)
        for individual in mutated:
            print(f'mutated chromosome : {individual.chromosome}')
            

        population = copy.deepcopy(new_population + mutated)

        print(f"\ngeneration({generation})")
        # Terminate condition
        total_objective = 0
        total_fitness = 0
        best_ingen = Individual([0, 0, 0, 0])
        best_ingen.objective = float('inf')
        for indiv in population:
            objective = indiv.objective_function(cost_matrix)
            fitness = indiv.fitness_function(objective)

            if objective < best_ingen.objective:
                best_ingen = indiv

            total_objective += objective
            total_fitness += fitness

            print(f'chromosome : {indiv.chromosome}, objective : {indiv.objective}, fitness : {indiv.fitness}') 
            if objective < best_solution.objective:
                best_solution = indiv
                best_solution_gen = generation
            if objective <= OBJECTIVE_THRESHOLD or fitness == 1:
                solution_found = True

        generation += 1
        result_bestIngen.append(best_ingen.fitness)
        result_objective.append((total_objective/POPULATION_SIZE))
        result_fitness.append((total_fitness/POPULATION_SIZE))
        print(f"result fitness size : {len(result_fitness)}")
        print(f"result objective size : {len(result_objective)}")
        
        print(generation)
        if solution_found or generation > GENERATION_THRESHOLD:
            end = time.time()
            break
    print(f'best chromosome : {best_solution}, objective : {best_solution.objective}, fitness : {best_solution.fitness} at generation {best_solution_gen}') 
    
    print(f'Execution Time : {end-start} second(s)')
    # Create a figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

    # Plotting Average Objective
    ax1.plot(result_objective, label='Average Objective', color='blue')
    ax1.set_ylabel('Objective')
    ax1.set_xlabel('Generation')
    ax1.legend()

    # Plotting Average Fitness
    ax2.plot(result_fitness, label='Average Fitness', color='red')
    ax2.plot(result_bestIngen, label='Best Fitness in Each Generation', color='green')
    ax2.annotate(
    text=f'generation {best_solution_gen}, fitness: {round(best_solution.fitness, 5)}, objective: {best_solution.objective}',
    xy=(best_solution_gen, best_solution.fitness),
    # xytext=(best_solution_gen+1, best_solution.fitness+1),
    # textcoords='offset points',
    arrowprops=dict(arrowstyle="->", color='black'))
    ax2.scatter(x = best_solution_gen, y = best_solution.fitness, label='Best solution', color='pink')
    ax2.set_ylabel('Fitness')
    ax2.set_xlabel('Generation')
    ax2.legend()

    plt.tight_layout()
    plt.show()