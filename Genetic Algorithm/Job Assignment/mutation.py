'''
This file contains mutation operation for job assignment GA loop
'''
from Individual import Individual
import random
import copy
def swap_gene(individual :Individual) -> Individual:
    chromosome = copy.deepcopy(individual.chromosome)
    index_list = range(len(chromosome))
    index1, index2 = random.sample(index_list, 2)
    # print(f"chromosome {individual} swapping index {index1}<->{index2} index value {individual.chromosome[index1]}<->{individual.chromosome[index2]}")
    chromosome[index1], chromosome[index2] = chromosome[index2], chromosome[index1]
    return Individual(chromosome)

def swap_mutation(parents :list[Individual], mutationRate :float) -> list[Individual]:
    mutated :list[Individual] = []
    for parent in parents:
        mutation_probability = random.uniform(0, 1)
        if mutation_probability < mutationRate:
            # print(f'mutate chromosome : {parent}, prob {mutation_probability}')
            mutated.append(swap_gene(parent))
        else:
            mutated.append(parent)
            pass
    return mutated

if __name__ == "__main__":
    seq = [1, 2, 3, 4]
    idx = range(len(seq))
    index1, index2 = random.sample(idx, 2)
    # print(f'idx : {idx}')
    # print(f'before : {seq}')
    # print(f'index to swapped : {index1} {index2}')
    seq[index1], seq[index2] = seq[index2], seq[index1]
    # print(f'after : {seq}')