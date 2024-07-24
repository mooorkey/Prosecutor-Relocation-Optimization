import random
import csv
from Individual import Worker


def generate_job_pref(preference_size):
    job_preferences = []
    shuffled_jobid = [jobid for jobid in range(1, preference_size+1)]
    random.shuffle(shuffled_jobid)
    for index, jobid in enumerate(shuffled_jobid):
        job_preference = [jobid, 30-index, random.randint(1, 6)]
        job_preferences.append(job_preference)
    return job_preferences
def generate_worker_data(worker_id, preference_size):
    return [
        worker_id, # Seniority no 
        random.randint(1, 6),  # Rank
        random.randint(1, 3),  # Job
        generate_job_pref(preference_size),
        random.randint(0, 1)  # Random assignment
    ]

def generate_job_data(job_id):
    return [
        job_id,
        f"Job{job_id}",
        f"Pos{job_id}",
        [random.randint(0, 10) for _ in range(6)]
    ]

def randomize_ja_data(worker_size, job_size, pref_size):
    # Generate worker data
    worker_datas = [generate_worker_data(i, pref_size) for i in range(worker_size)]
    # Generate job data
    job_datas = [generate_job_data(i) for i in range(1, job_size+1)] # the size of jobs must be more >= preferences size to prevent index out of range
    # job_datas = [generate_job_data(i) for i in range(1, 200)] # the size of jobs must be more >= preferences size to prevent index out of range
    return worker_datas, job_datas

def to_csv(data):
    with open('file.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['seniority number', 'rank', 'job', 'preferences'])
        for worker in data:
            writer.writerow([worker])


if __name__ == "__main__":
    worker_datas, job_datas = randomize_ja_data(10, 10, 5)
    for worker in worker_datas:
        print(worker)

    for job in job_datas:
        print(job)

    to_csv([Worker(*worker) for worker in worker_datas])

    
    # print(generate_job_pref(10))