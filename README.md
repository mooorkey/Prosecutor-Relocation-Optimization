# Prosecutor Relocation Optimization
This repository contains a journey of my graduation/final year project called "Prosecutor Relocation Optimization".
Basically, it is a job assignment problem but with additional constraints. 
To solve the problem, I choose 2 popular optimization algorithms including "Branch and Bound" and "Genetic Algorithm" to study, compare, and optimize.

# Job assignment problem
### Problem Definition
The job assignment problem involves assigning a set of tasks or jobs to a set of agents such that to total cost of assignments is minimized while satisfying certain constraints. 
### Cost Matrix
The cost matrix is a key component in this problem. It determines the cost associated with assigning each worker to each job

<p align="center">
  <img width="302" alt="image" src="https://github.com/user-attachments/assets/a84d9ba4-fc00-40a1-bfc7-08ae1f0a2740">
</p>
<p align="center">
	<em>Cost Matrix</em>
</p>

In the above matrix:
- Rows represent workers
- Columns represent jobs
- Each cell **(n, m)** represents the cost of assigning worker **n** to job **m**

### Example
```
cost_matrix4 = [
    [9, 2, 7, 8], 
    [6, 4, 3, 7], 
    [5, 8, 1, 8], 
    [7, 6, 9, 4]  
]
```
#### For the cost matrix above
- The cost of assigning **Worker 0** to **Job 0** is **9**.
- The cost of assigning **Worker 1** to **Job 2** is **3**.

#### Constraints:
1. One Job per Worker: Each worker can be assigned to only one job.
2. One Worker per Job: Each job must be assigned to exactly one worker.
#### Objective: 
Find the optimal assignment of workers to jobs that minimizes the total cost while satisfying the constraints above.


### Methods
#### Branch and Bound
Branch and Bound is a method for solving optimization problems by exploring feasible solutions. it divides the problem into smaller subproblems and uses bounding techniques to cut the branches that cannot provide an optimal solution.

<p align='center'>
  <img width="500" alt="image" src="https://github.com/user-attachments/assets/1d7454f7-b55b-4519-92b9-5d9d37fcd24b">
</p>
<p align="center">
	<em>Branch and Bound Procedure</em>
</p>

- **Approach:** Explores all possible assignments but eliminates suboptimal solutions to reduce computation, resulting in reduced execution time.
- **Limitation:** Due to its nature, it is most likely a semi-brute-force approach. It can be inefficient for large-scale problems.
- **Result:** Proved inefficient for large-scale problems with a high number of workers and jobs because the execution time or time complexity grows exponentially with the size of the problem.
<p align="center">
  <img width="388" alt="image" src="https://github.com/user-attachments/assets/95634c7c-5902-4837-b436-95f7984e4acf">
</p>
<p align="center">
	<em>Branch and Bound Time Complexity</em>
</p>



#### Genetic Algorithm
The genetic algorithm is an evolutionary optimization technique inspired by natural selection. It evolves a population of candidate solutions over generations to find an optimal or in most cases near-optimal solution.
<p align='center'>
  <img width="388" alt="image" src="https://github.com/user-attachments/assets/b2ed2744-a450-40f8-b5e8-60b57db575c4">
</p>
<p align="center">
	<em>Genetic Algorithm Procedure</em>
</p>

- **Approach:** The picture above shows the main cycle of the Genetic Algorithm including 
  - Initial Population: The algorithm begins with a randomly generated set of assignments.
  - Selection: Each assignment is evaluated based on its fitness, and the best-performing assignments are selected. In this case, **Roulette Wheel Selection** is used, where individuals are selected probabilistically based on their fitness.
  - Crossover: Selected assignments are combined to produce new candidates. **PMX (Partially Mapped Crossover)** is used, where two crossover points are chosen, and segments between these points are exchanged between parent solutions while preserving the relative ordering and position of genes.
  - Mutation: Random changes are applied to some assignments to introduce diversity and expand the solution space. **Swap Mutation** is used to prevent gene duplication within chromosomes, which could violate the constraint. In Swap Mutation, two genes are randomly selected and swapped to maintain valid assignments.
  - Iteration: This process of selection, crossover, and mutation is repeated over multiple generations until a termination condition is met, such as a maximum number of generations or achieving a satisfactory solution.
- **Result:** The Genetic Algorithm is effective for exploring a large solution space and can efficiently find good solutions. 
-  TODO: Add result pictures of exec time vs input size
# Prosecutor Relocation Problem(Constraint Job Assignment Problem)
### Problem Definition
The prosecutor relocation problem involves assigning jobs to workers(prosecutors) while maximizing the satisfaction/appraisal score and maintaining the satisfaction of capacity constraint. Each worker has their own list of job preferences. Each job has their own rank and each rank has thier capacity that must not be exceeded.
### Prosecutor Relocation's Cost Matrix
```
### The data
worker_datas = [
        # PriorityNo, Rank, Job
        [1, 5, 1, [ # worker 1
            # JobId, Score, JobRank
            [2, 30, 5],
            [3, 29, 5],
            [4, 28, 5]
        ], 0], # Relocation Type 1 -> Promotion, 0 -> Relocation
        [2, 5, 1, [ # worker 2
            [2, 30, 6],
            [3, 29, 6],
            [4, 28, 6]
        ], 1],
etc...
```
This above matrix is what the cost matrix of prosecutor relocation look like and structured as follows:
- Priority Number: Indicate which worker should be assigned to a job first(1 is highest).
- Rank: The current rank of the worker(maximum rank of 6).
- Job: The jobId of worker current job.
- Job Preferences List:
  - JobId: jobId of preferred job.
  - Score: Appraisal/Satisfaction score(maximum score of 30, non-redundant).
  - JobRank: The required rank of preferred job(maximum rank of 6).
- Relocation Type: 0 for Normal Relocation, 1 for Promotion. We have to assign job to worker who's promoting first(assign to worker relocType 1 before 0).
```
job_datas = [
#  jobId, Name, Pos, [rank 1 ------------>6]
    [1, "Job1", "Pos1", [6, 5, 4, 1, 2, 1]],
    [2, "Job2", "Pos2", [1, 2, 3, 69, 5, 6]],
    [3, "Job3", "Pos3", [1, 0, 1, 1, 0, 1]],
    [4, "Job4", "Pos4", [1, 1, 0, 1, 0, 0]],
    [5, "Job5", "Pos5", [0, 1, 0, 1, 1, 0]],
    [6, "Job6", "Pos6", [1, 1, 1, 0, 0, 1]],
    [7, "Job7", "Pos7", [0, 0, 1, 1, 1, 0]],
    [8, "Job8", "Pos8", [1, 1, 0, 0, 1, 1]],
    [9, "Job9", "Pos9", [1, 1, 1, 1, 0, 0]],
    [10, "Job10", "Pos10", [0, 1, 1, 0, 0, 1]],
    [11, "Job11", "Pos11", [1, 0, 1, 0, 1, 1]],
    [12, "Job12", "Pos12", [0, 0, 0, 0, 0, 0]],
    [13, "Job13", "Pos13", [0, 0, 0, 0, 0, 0]]
]
```

