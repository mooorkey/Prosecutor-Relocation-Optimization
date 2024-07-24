import copy
class WorkerJobPreference():
    def __init__(self, jobid, score, rank, eligible = False) -> None:
        self.jobid :int = jobid
        self.score :int = score
        self.rank :int = rank
        self.eligible :bool = eligible

    
    def __repr__(self) -> str:
        return f'[id: {self.jobid}, rank: {self.rank}, score: {self.score}]'

class Worker:
    def __init__(self, senioritynumber, currentrank, currentjobid, scores, reloctype) -> None:
        self.senioritynumber :int = senioritynumber
        self.currentrank :int = currentrank
        self.currentjob :int = currentjobid
        self.scores :list[WorkerJobPreference] = [WorkerJobPreference(*score) for score in scores] # appraisals score for each job
        self.reloctype :list = reloctype # if relocate => 0, if promotion => 1 
    def __repr__(self) -> str:
        return f'W{self.senioritynumber}J{self.currentjob}R{self.currentrank}'

class Job: 
    def __init__(self, jobid, jobname, positionname, capacitylist) -> None:
        self.jobid :int = jobid
        self.name :str = jobname
        self.position :str = positionname
        self.capacitylist :list[int] = [cap for cap in capacitylist]
        # print(capacitylist)
        # self.capacitylist :list[int] = copy.deepcopy(capacitylist)
        # self.capacitylist :list[int] = capacitylist # Shallow
    
    def getcap(self, rank):
        return self.capacitylist[rank-1]

    def __repr__(self) -> str:
        return f'J{self.jobid}C{self.capacitylist}'
    
class Gene(WorkerJobPreference):
    def __init__(self, jobid :int, score :int, rank :int, worker :Worker) -> None:
        super().__init__(jobid, score, rank)
        self.worker :Worker = worker
    
    def __repr__(self) -> str:
        return f"J{self.jobid}R{self.rank}S{self.score}"
        
class Individual():
    def __init__(self, chromosome :list[Gene], jobDatas :list[Job], objective :int = None) -> None:
        self.chromosome :list[Gene] = chromosome
        self.fitness :float = None
        self.objective :int = objective
        self.probability :float = None
        self.cumulative_probability :float = None
        self.jobDatas :list[Job] = jobDatas

    def __repr__(self) -> str:
        return f"{self.chromosome}"