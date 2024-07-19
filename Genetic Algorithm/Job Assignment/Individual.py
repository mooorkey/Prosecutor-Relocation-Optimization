class Individual:
    def __init__(self, chromosome :list) -> None:
        self.chromosome :list = chromosome
        self.objective = None
        self.fitness = None
        self.probability = None
        self.cumulative_probability = None
        self.crossoverRate = None

    def __repr__(self) -> str:
        return f"{self.chromosome}"
    
    def objective_function(self, cost_matrix :list[list]) -> int:
        cost = 0
        for worker, job in enumerate(self.chromosome):
            cost += cost_matrix[worker][job]
        self.objective = cost
        return cost
    
    def fitness_function(self, objective_function :int) -> float:
        fitness = 1/(1 + objective_function)
        self.fitness = fitness
        return fitness
