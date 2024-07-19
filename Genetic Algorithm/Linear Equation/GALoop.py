'''
This file contains a fully working Genetic Algorithm loop for solving
a linear equation "2a+4b+3c-2d=41"

The fitness evaluation function was hardcoded so remember to change if needed 

Selection Method: Roulette Wheel Selection
Crossover Method: 1 Point Crossover
Mutation Method: Randomly Mutate 

I've left most of the debugging parts (print statements and comments) 
in the code to demonstrate how this algorithm works. If you want to 
test the execution time, you may want to turn off all the print statements.

This code will run until the break condition is met
'''
import random
import matplotlib.pyplot as plt
import copy

class individual:
    def __init__(self, chromosome :list) -> None:
        self.chromosome = chromosome
        self.fitness = self.fitness_function(self.chromosome)
        self.probability = None # roulette wheel selection
        self.cumulative = None # roulette wheel selection
        self.crossoverRate = None
        self.objective = self.objective_function(self.chromosome)

    def __repr__(self) -> str:
        return f"{self.chromosome}"

    @staticmethod
    def fitness_function(param :list):
        # print(f"2*{param[0]}+4*{param[1]}+3*{param[2]}-2*{param[3]}")
        obj = abs((2*param[0]+4*param[1]+3*param[2]-2*param[3])-41) # g(x)
        fitness_value = 1/(1 + obj)
        return fitness_value
    
    @staticmethod
    def objective_function(param :list):
        objective_function = abs((2*param[0]+4*param[1]+3*param[2]-2*param[3])-41) # g(x)
        return objective_function

def crossover(parents :list[individual], crossoverRate :float) -> list[individual]:
    children :list[individual] = []
    selected_parents :list[individual] =[] # lsit of to mate parents
    for index in range(len(parents)): # Random Crossover Rate for Each parent Chromosome
        parents[index].crossoverRate = random.uniform(0 ,1)
        print(f"chromosome : {parents[index].chromosome}, fitness : {parents[index].fitness}, crsoRate : {parents[index].crossoverRate}")
        if parents[index].crossoverRate > crossoverRate: # if parent has a chance then added to to_mate list
            print(f"{parents[index].crossoverRate} > {crossoverRate} {parents[index].crossoverRate > crossoverRate} satisfied")
            selected_parents.append(parents[index])
        else: # if parent doesn't has a chance then dont mate
            children.append(parents[index])
            print(f"{parents[index].crossoverRate} > {crossoverRate} {parents[index].crossoverRate > crossoverRate} not satisfied")
        print("")
    print(f"\nSelected Parents : {selected_parents}")
    print(f"Not Selected Parents : {children}\n")
    if len(selected_parents) > 1:
        for index, parent in enumerate(selected_parents):
            next = index + 1
            if next >= len(selected_parents):
                next = 0
            next_parent = selected_parents[next]
            crossoverPoint = random.randint(1, len(selected_parents)-1)
            child = parent.chromosome[:crossoverPoint] + next_parent.chromosome[crossoverPoint:]
            print(f"crossoverPoint {crossoverPoint} {parent}x{next_parent} = {parent.chromosome[:crossoverPoint]} + {next_parent.chromosome[crossoverPoint:]} = {child}")
            children.append(individual(child))
    elif len(selected_parents) == 1:
        children.append(selected_parents.pop())
    return children

def mutation(parents :list[individual], mutationRate :float) -> list[individual]:
    total_gene = POPULATION_SIZE*CHROMOSOME_SIZE #
    total_mutated = total_gene*mutationRate
    print(f"\ntotal gene {total_gene} mutated {total_mutated} gene({round(total_mutated)})")
    print(f"accepted parent : {parents}")
    for _ in range(round(total_mutated)):
        random_index = random.randint(0, total_gene-1)
        row = random_index // 4 # get chromosome index
        col = random_index % 4 # get gene index
        random_gene = random.randint(-41, 41)
        print(f"\nmutated index : {random_index} ({row},{col}), random gene : {random_gene}")
        old_chromosome = copy.deepcopy(parents[row].chromosome)
        print(f"from parent {old_chromosome}", end=" ")
        old_chromosome[col] = random_gene
        new_parent = individual(old_chromosome)
        parents[row] = new_parent
        print(f"-> {parents[row]}")
    print(f"mutated {parents}")
    print("\n")
    return parents

    

if __name__ == "__main__":

    result_objective :list = []
    result_fitness :list = []
    result_bestIngen :list = []

    POPULATION_SIZE = 4
    CHROMOSOME_SIZE = 4
    CHROMOSOME_RANGE = 41

    CROSSOVER_RATE = 0.25
    MUTATION_RATE = 0.2
    
    population = []
    solution_found = False
    error_found = False
    error = 0
    # Random Initail Population
    for _ in range(POPULATION_SIZE):
        individual_object = individual([random.randint(0, CHROMOSOME_RANGE) for _ in range(CHROMOSOME_SIZE)])
        population.append(individual_object)
    generation = 0

    while True:
        total_fitness = 0
        cumulative_probability = 0
        # getting the total fitness
        for object in population:
            print(f"chromosome : {object.chromosome}, fitness : {object.fitness}, objective : {object.objective}")
            total_fitness += object.fitness
            
        print("")   
        # getting the cumulative probability of each individual
        for object in population:
            object.probability = object.fitness / total_fitness
            cumulative_probability += object.probability
            object.cumulative = cumulative_probability
            print(f"chromosome : {object.chromosome}, fitness : {object.fitness}, probability : {object.probability}, cumulative proability : {object.cumulative}")

        print("")   
        selected_parents = []

        while len(selected_parents) < POPULATION_SIZE:
            random_Number = random.uniform(0, 1)
            print(f"Random Number : {random_Number}")
            for object in population:
                if (random_Number <= object.cumulative) and len(selected_parents) < POPULATION_SIZE:
                    selected_parents.append(object)
                    print(f"Selected Parent {object.chromosome}, cumulative probability : {object.cumulative}")
                    break
        print("")   
        
        mated = crossover(selected_parents, CROSSOVER_RATE)
        print(f"\ncrossover output : {mated}\n")   
    
        mutated = mutation(mated, MUTATION_RATE)

        for indiv in mutated:
            print(f"chromosome(mutated){indiv.chromosome}")

        population = copy.deepcopy(mutated)
        print(f"\nEnd Gen {generation}\n\nReporting Result")
        generation += 1

        total_fitness = 0
        total_objective = 0
        for indiv in population:
            print(f"Chromosome : {indiv.chromosome}, objective function : {indiv.objective}, fitness {indiv.fitness}")
            total_fitness += indiv.fitness
            total_objective += indiv.objective
            if indiv.objective == 0:
                solution = (2*indiv.chromosome[0]+4*indiv.chromosome[1]+3*indiv.chromosome[2]-2*indiv.chromosome[3])-41
                print(f"\nFound Result a={indiv.chromosome[0]}, b={indiv.chromosome[1]}, c={indiv.chromosome[2]}, a={indiv.chromosome[3]}")
                print(f"2*{indiv.chromosome[0]}+4*{indiv.chromosome[1]}+3*{indiv.chromosome[2]}-2*{indiv.chromosome[3]}-41=", end="")
                print(f"{solution} abs {abs(solution)}\n")
                solution_found = True
                if indiv.objective != solution:
                    error += 1
                    error_found = True
        result_objective.append((total_objective/POPULATION_SIZE))
        result_fitness.append((total_fitness/POPULATION_SIZE))
        
        # break condition
        if solution_found or error_found or generation > 1000:
            if error_found:
                print("Error : ", error)
            break


