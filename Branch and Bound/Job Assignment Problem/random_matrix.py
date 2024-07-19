import random
import time

def rnd_mat(n):
    matrix = []
    random.seed(time.time())
    row = []
    for i in range(n):
        for j in range(n):
            row.append(random.randrange(0, n))
        matrix.append(row)
        row = []
            
    return matrix

'''
This function generated a random matrix with the size nxn
'''
def generate_matrix(n):
    matrix = [0] * n
    numbers = list(range(1, n + 1))

    for i in range(n):
        random.shuffle(numbers)
        matrix[i] = numbers[:]

    return matrix

def print_matrix(matrix):
    for row in matrix:
        print(" ".join(map(str, row)))

if __name__ == "__main__":
    # mat = rnd_mat(10)
    # print(f"NxN = {len(mat)}x{len(mat[0])}")
    # print(mat)
    start = time.time()
    mat = generate_matrix(5)
    end = time.time()
    print(f"nxn : {len(mat)}x{len(mat[0])}")
    print("Exec Time :", end-start, "s")
    print(mat)