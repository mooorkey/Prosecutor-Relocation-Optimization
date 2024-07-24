'''
This file contains crossover operation used in GA loop
I got the reference from https://github.com/DEAP/deap/blob/master/deap/tools/crossover.py
and code my own based on that reference for my own understanding
'''
import random
from Individual import Individual
import copy
import numpy as np
# Reference
def cxPartialyMatched(ind1, ind2):
    # print(f'input : {ind1} {ind2}')
    size = min(len(ind1), len(ind2))
    p1, p2 = [0] * size, [0] * size

    # Initialize the position of each indices in the individuals
    for i in range(size):
        p1[ind1[i]] = i
        p2[ind2[i]] = i

    # print(f'\np list : {p1} {p2}\n')
    # Choose crossover points
    cxpoint1 = 3
    cxpoint2 = 7
    # cxpoint1 = random.randint(0, size)
    # cxpoint2 = random.randint(0, size - 1)
    if cxpoint2 >= cxpoint1:
        cxpoint2 += 1
    else:  # Swap the two cx points
        cxpoint1, cxpoint2 = cxpoint2, cxpoint1

    # Apply crossover between cx points
    for i in range(cxpoint1, cxpoint2):
        # Keep track of the selected values
        temp1 = ind1[i]
        temp2 = ind2[i]
        # Swap the matched value
        ind1[i], ind1[p1[temp2]] = temp2, temp1
        ind2[i], ind2[p2[temp1]] = temp1, temp2
        # Position bookkeeping
        p1[temp1], p1[temp2] = p1[temp2], p1[temp1]
        p2[temp1], p2[temp2] = p2[temp2], p2[temp1]
    return ind1, ind2

def pmx_crossover_prototype(parents :list[Individual], crsoRate :float) -> list[Individual]: 
    selected_parents :list[Individual] = []
    children :list[Individual] = []
    POPULATION_SIZE = len(parents)
    CHROMOSOME_SIZE = len(parents[0].chromosome)
    # print(f"{POPULATION_SIZE}x{CHROMOSOME_SIZE}")

    # Random crossover rate for each parent
    # print("Randomizing parents crossover rate")
    for parent in parents:
        parent.crossoverRate = random.uniform(0 ,1)
        # print(f"parent : {parent}, crossover rate : {parent.crossoverRate}")
        if parent.crossoverRate < crsoRate:
            selected_parents.append(parent)
        else:
            children.append(parent)


    # Perform PMX
    if len(selected_parents) > 1:
        # print(f"selected parent(s) : {selected_parents}")
        # print(f"not selected parent(s) : {children}")
        for index in range(len(selected_parents)):
            next_index = index + 1
            if next_index > len(selected_parents) - 1:
                next_index = 0

            # TODO Create seperate pmx that take 2 parent as input instead of a list of parent to prevent mate the same parent twice
            # and implement pairing logic if odd then there will be 1 parent that can't be made and directly be a offspring for next generation
            parent1 = selected_parents[index]
            parent2 = selected_parents[next_index]
            crossover_point1 = random.randint(0, CHROMOSOME_SIZE - 1)
            crossover_point2 = random.randint(0, CHROMOSOME_SIZE - 1)
            # crossover_point1 = 3 # for debugging
            # crossover_point2 = 8 # for debugging
            if crossover_point1 > crossover_point2:
                crossover_point1, crossover_point2 = crossover_point2, crossover_point1
            # print(f"mate {parent1} <=> {parent2}, crossover slice index {crossover_point1} -> {crossover_point2}")


            # A list that keep tracking positions of genes in ind1 and ind2
            parent1_gene_position = {val: i for i, val in enumerate(parent1.chromosome)}
            parent2_gene_position = {val: i for i, val in enumerate(parent2.chromosome)}

            # print(f'position mapping list : {parent1_gene_position} {parent2_gene_position}')

            offspring1, offspring2 = [0] * CHROMOSOME_SIZE, [0] * CHROMOSOME_SIZE
            offspring1 = copy.deepcopy(parent1.chromosome)
            offspring2 = copy.deepcopy(parent2.chromosome)
            
            for index in range(crossover_point1, crossover_point2 + 1):
                # print(f'\tcrossover point : {index}')
                # print(f'\tfrom offspring : {offspring1} {offspring2}')
                gene1 = offspring1[index]
                gene2 = offspring2[index]
                # print(f'\tcrossover index : {parent1_gene_position[gene2]}<->{parent2_gene_position[gene1]}, swapping value {gene1}<->{gene2}')

                # Swap Matched Value
                offspring1[index], offspring1[parent1_gene_position[gene2]] = gene2, gene1
                offspring2[index], offspring2[parent2_gene_position[gene1]] = gene1, gene2

                # Update Gene Position
                parent1_gene_position[gene1], parent1_gene_position[gene2] = parent1_gene_position[gene2], parent1_gene_position[gene1]
                parent2_gene_position[gene1], parent2_gene_position[gene2] = parent2_gene_position[gene2], parent2_gene_position[gene1]

                # print(f'\tupdated book keeping : {parent1_gene_position} {parent2_gene_position}')
                # print(f'\tto offspring : {offspring1} {offspring2}\n')

            # print(f'final crossovered offspring : {offspring1} {offspring2}\n')
            children.append(Individual(offspring1))
            children.append(Individual(offspring2))

    elif len(selected_parents) == 1:
        # print("not enough parent to mate")
        children.append(selected_parents.pop())
    else:
        # print("no parent have been selected")
        pass
    return children

def pmx(parent1 :Individual, parent2 :Individual) -> list[Individual]:
    CHROMOSOME_SIZE = len(parent2.chromosome)
    crossover_point1 = random.randint(0, CHROMOSOME_SIZE - 1)
    crossover_point2 = random.randint(0, CHROMOSOME_SIZE - 1)
    # crossover_point1 = 3 # for debugging
    # crossover_point2 = 8 # for debugging
    if crossover_point1 > crossover_point2:
        crossover_point1, crossover_point2 = crossover_point2, crossover_point1
    print(f"mate {parent1} <=> {parent2}, crossover slice index {crossover_point1} -> {crossover_point2}")


    # A list that keep tracking positions of genes in ind1 and ind2
    parent1_gene_position = {val: i for i, val in enumerate(parent1.chromosome)}
    parent2_gene_position = {val: i for i, val in enumerate(parent2.chromosome)}

    print(f'position mapping list : {parent1_gene_position} {parent2_gene_position}\n ')

    offspring1, offspring2 = [0] * CHROMOSOME_SIZE, [0] * CHROMOSOME_SIZE
    offspring1 = copy.deepcopy(parent1.chromosome)
    offspring2 = copy.deepcopy(parent2.chromosome)
    
    for index in range(crossover_point1, crossover_point2 + 1):
        print(f'\tcrossover point : {index}')
        print(f'\tfrom offspring : {offspring1} {offspring2}')
        gene1 = offspring1[index]
        gene2 = offspring2[index]
        if (gene1 not in parent2_gene_position) or (gene2 not in parent1_gene_position):
            print('Key not found -> Skipping this index')
            continue
        print(f'\tcrossover index : {parent1_gene_position[gene2]}<->{parent2_gene_position[gene1]}, swapping value {gene1}<->{gene2}')

        # Swap Matched Value
        offspring1[index], offspring1[parent1_gene_position[gene2]] = gene2, gene1
        offspring2[index], offspring2[parent2_gene_position[gene1]] = gene1, gene2

        # Update Gene Position
        parent1_gene_position[gene1], parent1_gene_position[gene2] = parent1_gene_position[gene2], parent1_gene_position[gene1]
        parent2_gene_position[gene1], parent2_gene_position[gene2] = parent2_gene_position[gene2], parent2_gene_position[gene1]

        print(f'\tupdated book keeping : {parent1_gene_position} {parent2_gene_position}')
        print(f'\tto offspring : {offspring1} {offspring2}\n')

    offspring1, offspring2 = Individual(offspring1), Individual(offspring2)
    print(f'final offspring : {offspring1.chromosome} {offspring2.chromosome}\n')
    return offspring1, offspring2

def pair_elements(test_list) -> list:
        return [[test_list[i], test_list[i + 1]] if i + 1 < len(test_list) else [test_list[i]] for i in range(0, len(test_list), 2)]

def pmx_crossover(parents :list[Individual], crsoRate :float) -> list[Individual]: 
    selected_parents :list[Individual] = []
    children :list[Individual] = []

    # Random crossover rate for each parent
    # print(f"Randomizing parents crossover rate({crsoRate})")
    for parent in parents:
        parent.crossoverRate = random.uniform(0 ,1)
        # print(f"parent : {parent}, crossover rate : {parent.crossoverRate}")
        if parent.crossoverRate < crsoRate:
            selected_parents.append(parent)
        else:
            children.append(parent)

    paired_parents :list[list[Individual]] = pair_elements(selected_parents)
    parent_number = len(selected_parents)
    if parent_number > 1:
        for parent_pair in paired_parents:
            if len(parent_pair) < 2:
                parent = parent_pair
                # print(f'no pair parent : {parent}')
                children.append(parent[0])
            else:
                parent1, parent2 = parent_pair
                print(f'crossover parent : {parent1}<=>{parent2}')
                offspring1, offspring2 = pmx(parent1, parent2)
                children.append(offspring1)
                children.append(offspring2)
    elif parent_number == 1:
        # print("not enough parent to mate")
        children.append(parents.pop())
    else:
        # print("no parent have been selected")
        return parents

    return children

# Driver Code
if __name__ == "__main__":
    parent1 = Individual([1,2,3,4,5,6,7,8,9])
    parent2 = Individual([5,4,6,7,2,1,3,9,8])
    parent3 = Individual([1,3,5,7,9,2,4,6,8])
    parent4 = Individual([1,4,7,2,5,8,3,6,9])
    parent5 = Individual([3,1,2,4,9,7,8,6,5])
    # pmx_crossover_prototype([parent1, parent2], 25)

    # pmx(parent1, parent2)
    pmx_crossover([parent1, parent2, parent3, parent4, parent5], 25)

    # p1 = [0, 1, 2, 3, 4 ,5 ,6 ,7, 8]
    # p2 = [4, 3, 5, 6, 1, 0, 2, 8, 7]
    # o1, o2 = cxPartialyMatched(p1, p2)
    # print(o1, o2)


