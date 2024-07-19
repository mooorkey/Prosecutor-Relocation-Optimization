'''
This is the base code used to 
study branch and bound and 
solve the common job assignment problem
'''

input_data = [ # the cost matrix
#job 0, 1, 2, 3 
    [9, 2, 7, 8], # Worker 0
    [6, 4, 3, 7], # Worker 1
    [5, 8, 1, 8], # Worker 2
    [7, 6, 9, 4]  # Worker 3
]

def calculate_cost(matrix, assignment):
    cost = 0
    for i, j in enumerate(assignment):
        cost += matrix[i][j] # Add cost of assignment
    return cost

def find_least_cost(cost_matrix): # The branch and bound function
    worker_length = len(cost_matrix)

    # Initialize with positive infinity
    final_cost = float('inf') 

    # result matrix
    assignment_result = [] 

    # Initialize stack
    stack = [(0, [], list(range(worker_length)))] 
    print(stack, "\n")
    while stack:

        # cost, list of assigned worker, list of unassigned worker
        cost, assignment, unassigned = stack.pop() # pop a node to explore

        if not unassigned: # Check if the unassigned list is empty (empty = all worker are assigned to a job)
            if cost < final_cost: # if the assignment cost is less than final cost then it should be a result
                final_cost = cost
                assignment_result = assignment
            continue

        # create child node for each unassigned worker
        for job in unassigned:
            print(f'creating a child node')
            new_assignment = assignment + [job] # add job to assignment
            new_unassigned = [j for j in unassigned if j != job] # create new unassigned list
            new_cost = calculate_cost(cost_matrix, new_assignment) # Evaluate node cost
            
            # check if the newly created node cost is more than final if so we don't need to explore (Terminate)
            if new_cost < final_cost: 
                stack.append((new_cost, new_assignment, new_unassigned))        

    return final_cost, assignment_result

        
solution, assignment = find_least_cost(input_data)
print("Optimal Solution:", solution)
print("Optimal Assignment:", assignment)