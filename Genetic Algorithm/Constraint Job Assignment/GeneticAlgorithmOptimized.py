'''
This file is the completed version of genetic algorithm that I used to solve constraint job assignment problem
it comes with the profiling tool used to analyze execution time of each function

in the datafile there are various size of data
'''
from Individual import Individual, Worker, Job, Gene, WorkerJobPreference
import random
import copy
import matplotlib.pyplot as plt
import time
import numpy as np
import cProfile
import pstats


from randomize_data import randomize_ja_data

from DataFile import worker_datas, job_datas, worker_datas2, worker_datas3, job_datas2, job_datas3

def pair_elements(list) -> list:
        return [[list[i], list[i + 1]] if i + 1 < len(list) else [list[i]] for i in range(0, len(list), 2)]

def fitness_function(individual :Individual) -> int:
    objective = 0
    for gene in individual.chromosome:
        objective += gene.score
    # fitness = 1 - (1/objective) 
    global MAX_SCORE
    fitness = (objective) / MAX_SCORE
    individual.objective = objective
    individual.fitness = fitness

    return objective, fitness

def mutation(parents :list[Individual], mutationRate: float) -> list[Individual]:
    # Perform Mutation
    mutated :list[Individual] = []
    for individual in parents:
        random_number = random.uniform(0 ,1)
        if random_number < mutationRate:
            # Do Mutation
            # chromosome = copy.deepcopy(individual.chromosome)
            chromosome = [Gene(gene.jobid, gene.score, gene.rank, gene.worker) for gene in individual.chromosome]
            gene_index = random.randint(0, CHROMOSOME_SIZE - 1) # Random Gene index
            gene = chromosome[gene_index]
            # preferencesTemp :list[WorkerJobPreference] = copy.deepcopy(gene.worker.scores)
            preferencesTemp :list[WorkerJobPreference] = [WorkerJobPreference(scr.jobid, scr.score, scr.rank) for scr in gene.worker.scores]
            mutated_gene :Gene = None
            # for jobpref in preferencesTemp:
            while preferencesTemp:
                random_jobIndex = random.choice(range(len(preferencesTemp)))
                random_job = preferencesTemp[random_jobIndex]
                prefered_job = next(filter(lambda job : job.jobid == random_job.jobid, individual.jobDatas))

                # Check for capacity
                if prefered_job.capacitylist[random_job.rank - 1] > 0:
                    old_job = next(filter(lambda job : job.jobid == gene.jobid, individual.jobDatas))
                    if prefered_job == old_job: # Prevent from duplication
                        preferencesTemp.pop(random_jobIndex)
                        continue
                    # Create Gene
                    mutated_gene = Gene(random_job.jobid, random_job.score, random_job.rank, gene.worker)

                    # Update Job Capacity
                    old_job.capacitylist[gene.rank - 1] += 1
                    prefered_job.capacitylist[random_job.rank - 1] -= 1
                    break
                else:
                    preferencesTemp.pop(random_jobIndex)
            if mutated_gene != None:
                chromosome[gene_index] = mutated_gene
                # mutated_individual = Individual(chromosome, copy.deepcopy(individual.jobDatas))
                mutated_individual = Individual(chromosome, [Job(job.jobid, job.name, job.position, [cap for cap in job.capacitylist]) for job in individual.jobDatas])
                objective, fitnesss = fitness_function(mutated_individual)
                mutated.append(mutated_individual)
            else:
                mutated.append(individual)
        else:
            mutated.append(individual)
    return mutated

def crossover(selected_parents :list[Individual], crossoverRate :float) -> list[Individual]:
    # Perform Crossover
    to_crossover :list[Individual] = []
    mated :list[Individual] = []
    # Radomizing Parent Crossover Rate
    for parent in selected_parents:
        crsoRate = random.uniform(0, 1)
        if crsoRate < crossoverRate:
            to_crossover.append(parent)
        else:
            mated.append(parent)
    
    if len(to_crossover) > 1:
        # Performing crossover
        paired_parents :list[list[Individual]] = pair_elements(sorted(to_crossover, key=lambda chromosome : chromosome.fitness, reverse=True))
        for pair in paired_parents:
            if len(pair) < 2:
                mated.append(pair[0])
            else:
                parent1, parent2 = pair
                global CHROMOSOME_SIZE
                crossover_point1 = random.randint(0, CHROMOSOME_SIZE - 1)
                crossover_point2 = random.randint(0, CHROMOSOME_SIZE - 1)
                if crossover_point2 < crossover_point1:
                    crossover_point1, crossover_point2 = crossover_point2, crossover_point1

                for index in range(crossover_point1, crossover_point2 + 1):
                    # Check before swapping
                    gene1 = parent1.chromosome[index]
                    gene2 = parent2.chromosome[index]
                    
                    # offspring1_chromosome = [c1 for c1 in parent1.chromosome] # <-- this way got the same ref/addr
                    # offspring1_jobDatas = [j1 for j1 in parent1.jobDatas]
                    # offspring2_chromosome = [c2 for c2 in parent2.chromosome]
                    # offspring2_jobDatas = [j2 for j2 in parent2.jobDatas]

                    offspring1_chromosome = [Gene(c1.jobid, c1.score, c1.rank, c1.worker) for c1 in parent1.chromosome]
                    offspring1_jobDatas = [Job(j1.jobid, j1.name, j1.position, j1.capacitylist) for j1 in parent1.jobDatas]
                    offspring2_chromosome = [Gene(c2.jobid, c2.score, c2.rank, c2.worker) for c2 in parent2.chromosome]
                    offspring2_jobDatas = [Job(j2.jobid, j2.name, j2.position, j2.capacitylist) for j2 in parent2.jobDatas]
                    

                    job1 = next(filter(lambda job : job.jobid == gene1.jobid, parent1.jobDatas))
                    job2 = next(filter(lambda job : job.jobid == gene2.jobid, parent2.jobDatas))

                    # Check capacity
                    if job1.capacitylist[gene2.rank - 1] > 0 and job2.capacitylist[gene1.rank - 1] > 0:
                        # Perform swapping
                        offspring1_chromosome[index], offspring2_chromosome[index] = gene2, gene1


                        # Update JobDatas
                        # Increase original job
                        offspring1_jobDatas[job1.jobid - 1].capacitylist[gene1.rank - 1] += 1
                        offspring2_jobDatas[job2.jobid - 1].capacitylist[gene2.rank - 1] += 1
                        # Reduce new job
                        offspring1_jobDatas[job2.jobid - 1].capacitylist[gene2.rank - 1] -= 1
                        offspring2_jobDatas[job1.jobid - 1].capacitylist[gene1.rank - 1] -= 1

                    else:
                        pass
                # Create Individual 
                offspring1 = Individual(offspring1_chromosome, offspring1_jobDatas)
                offspring2 = Individual(offspring2_chromosome, offspring2_jobDatas)
                mated.append(offspring1)
                mated.append(offspring2)

                # Calculating Fitness
                objective1, fitness1 = fitness_function(offspring1)
                objective2, fitness2 = fitness_function(offspring2)
    elif len(to_crossover) == 1:
        mated.append(to_crossover.pop())
    else:
        pass
        
    return mated

def generate_initial_population(size, workers_list: list[Worker], jobs_list: list[Job]):
    population :list[Individual] = []
    # Generating Initial Population
    for _ in range(size):
        chromosome :list[Gene] = []
        individual_jobs_list = [Job(jobt.jobid, jobt.name, jobt.position, [cap for cap in jobt.capacitylist]) for jobt in jobs_list]
        individual_workers_list = [Worker(workert.senioritynumber, workert.currentrank, workert.currentjob, [[score.jobid, score.score, score.rank, score.eligible] for score in workert.scores], workert.reloctype) for workert in workers_list]
        for worker in individual_workers_list:
            stop = False
            while not stop:
                eligible = list(filter(lambda job : job.eligible == True, worker.scores)) # --> get a list of eligible job pref
                chosen_job_pref = random.choice(eligible)
                chosen_job = next(filter(lambda job : job.jobid == chosen_job_pref.jobid, individual_jobs_list))
                if chosen_job.capacitylist[chosen_job_pref.rank - 1] - 1 >= 0: # if capacity constraint satisfy
                    # reduce capacity chosen job capacity
                    chosen_job.capacitylist[chosen_job_pref.rank - 1] -= 1
                    stop = True
                else:
                    # set eligible flag to false and random again
                    chosen_job_pref.eligible = False
            chromosome.append(Gene(chosen_job_pref.jobid, chosen_job_pref.score, chosen_job_pref.rank, worker))
        individual = Individual(chromosome, individual_jobs_list)
        objective, fitness = fitness_function(individual)
        population.append(individual)
    return population
def GA(worker_datas_i, job_datas_i, CRSO, MUT, ELT, POP, GEN, dbg):
    average_objective :list = []
    average_fitness :list = []
    result_bestIngen :list = []

    global MAX_SCORE
    MAX_SCORE = len(worker_datas_i) * 30

    # Packing data to object
    workers_list :list[Worker] = []
    jobs_list :list[Job] = []
    for worker in worker_datas_i:
        new_worker = Worker(*worker)
        workers_list.append(new_worker)
    for job in job_datas_i:
        new_job = Job(*job)
        jobs_list.append(new_job)

    # Clearing slot
    for worker in workers_list:
        # Finding worker current job
        worker_currentjob = next(filter(lambda job : job.jobid == worker.currentjob, jobs_list))
        # +1 capacity for worker's current job
        worker_currentjob.capacitylist[worker.currentrank - 1] += 1


    # Set Eligible Flag for each job pref
    for worker in workers_list:
        for job in worker.scores:
            prefered_job = jobs_list[job.jobid-1]
            if prefered_job.getcap(job.rank) > 0:
                job.eligible = True



    # GA Parameter
    POPULATION_SIZE = POP
    GENERATION_THRESHOLD = GEN
    CrossoverRate = CRSO
    MutationRate = MUT
    Elite_percentage = ELT
    generation = 0

    global CHROMOSOME_SIZE
    CHROMOSOME_SIZE = len(workers_list)
    population :list[Individual] = generate_initial_population(POPULATION_SIZE, workers_list, jobs_list)
    best_solution :Individual = Individual(None, None)
    best_solution.objective = -float('inf')
    best_solution_gen = -float('inf')

    
        

    while True:
        total_fitness = 0
        # Calculate fitness
        for index, indiv in enumerate(population):
            objective, fitness = fitness_function(indiv)
            total_fitness += fitness

        # Preserve elite chromosome over to the next generation
        population.sort(key=lambda x: x.fitness, reverse=True)
        Elite_count = int(POPULATION_SIZE * Elite_percentage)
        # Elite = copy.deepcopy(population[:Elite_count])
        Elite = [Individual(chrms.chromosome, chrms.jobDatas) for chrms in population[:Elite_count]]
        new_population = Elite
        
        
        # Calculating cumulative probability
        cumulative_probability = 0
        for index, individual in enumerate(population):
            individual.probability = individual.fitness / total_fitness
            cumulative_probability += individual.probability
            individual.cumulative_probability = cumulative_probability
        
        # Roulette Wheel Selection
        selected_parents :list[Individual] = []
        while(len(selected_parents) < POPULATION_SIZE - len(new_population)):
            random_number = random.uniform(0, 1)
            for individual in population:
                if random_number < individual.cumulative_probability:
                        selected_parents.append(individual)
                        break
        # Crossover
        offsprings :list[Individual] = crossover(selected_parents, CrossoverRate)
        
        # Mutation
        mutated :list[Individual]= mutation(offsprings, MutationRate)

        # Form new generation 
        new_population.extend(mutated)
        # population = copy.deepcopy(new_population)
        # population = [Individual(chrms.chromosome, chrms.jobDatas) for chrms in new_population]
        population = new_population


        # Terminate Condition
        total_fitness = 0
        total_objective = 0
        best_ingen = Individual(None, None)
        best_ingen.objective = -float('inf')
        generation += 1

        # Reporting
        for indiv in population:
            objective, fitness = fitness_function(indiv)

            if objective > best_ingen.objective:
                best_ingen = indiv

            total_objective += objective
            total_fitness += fitness

            # print(f'Chromosome : {indiv.chromosome}, objective : {indiv.objective}, fitness : {indiv.fitness}') 
            if objective > best_solution.objective:
                best_solution = indiv
                best_solution_gen = generation
        if dbg:
            print(f'\nGeneration {generation}')
            print(f'({len(workers_list)})Best Chromosome : {best_solution}')
            print(f'({len(workers_list)})objective : {best_solution.objective}({MAX_SCORE}), fitness : {best_solution.fitness} at generation {best_solution_gen}\n')
        result_bestIngen.append(best_ingen.fitness)
        average_objective.append((total_objective/POPULATION_SIZE))
        average_fitness.append((total_fitness/POPULATION_SIZE))
        if generation >= GENERATION_THRESHOLD:
            break
    print(f'({len(workers_list)})Best Chromosome : {best_solution}')
    print(f'({len(workers_list)})objective : {best_solution.objective}({MAX_SCORE}), fitness : {best_solution.fitness} at generation {best_solution_gen}\n')
    return best_solution, average_fitness, average_objective, result_bestIngen, best_solution_gen
    

if __name__ == "__main__":
    worker_size = 500
    job_size = 500 # must be >= pref_size
    pref_size = 30

    # GA PARAMETER

    CRSO = 0.43
    MUT = 0.8
    ELT = 0.2
    POP = 10
    GEN = 2000
    wdatas, jdatas = randomize_ja_data(worker_size, job_size, pref_size)
    start = time.perf_counter()
    # cProfile.run('GA(wdatas, jdatas, CRSO, MUT, ELT, POP, GEN, False)', "profilingResults.cprof")
    # cProfile.run('GA(worker_datas, job_datas, CRSO, MUT, ELT, POP, GEN, False)', "profilingResults.cprof")
    cProfile.run('GA(worker_datas2, job_datas2, CRSO, MUT, ELT, POP, GEN, False)', "profilingResults.cprof")
    # cProfile.run('GA(worker_datas3, job_datas3, CRSO, MUT, ELT, POP, GEN, False)', "profilingResults.cprof")
    end = time.perf_counter()
    exec_time = end - start
    print(f'Execution Time : {exec_time:.5f}s')
    with open("./Benchmarking/ResultListComp.txt", "w") as f:
        ps = pstats.Stats("profilingResults.cprof", stream=f)
        ps.sort_stats('cumulative')
        ps.print_stats()

    


