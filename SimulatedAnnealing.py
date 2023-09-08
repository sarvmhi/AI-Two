import random
import pysat
from pysat.formula import CNF
from pysat.solvers import Solver
import math

# two cnf files, one is satisfiable and the other is unsatisfiable
Uformula = CNF(from_file="UInput.cnf")
formula = CNF(from_file="Input.cnf")

# Create a solver instance for unsatisfiable formula
usolver = Solver()
usolver.append_formula(Uformula.clauses)

# Create a solver instance for satisfiable formula
solver = Solver()
solver.append_formula(formula.clauses)

# actually solves it and can find out if it's satisfiable
print(usolver.solve())
print(solver.solve())

# if it's satisfiable we can get an answer
print(usolver.get_model())
print(solver.get_model())

# we can make a test list and feed it to the model as an assumption
variables = []
# test for satisfiable formula:
for i in range(1, formula.nv+1):
    var = (random.randint(0, 1))
    if var == 1:
        variables.append(i)
    else:
        variables.append(-i)

print(solver.solve(assumptions=variables))

# define a function to evaluate the fitness of an individual
def fitness(individual):
    # count the number of clauses satisfied by the individual
    count = 0
    for clause in formula.clauses:
        # check if any literal in the clause is true under the individual
        satisfied = False
        for literal in clause:
            # get the index and sign of the literal
            index = abs(literal) - 1 # zero-based indexing
            sign = literal > 0 # positive or negative literal
            # get the value of the corresponding variable in the individual
            value = individual[index]
            # check if the literal and the value match
            if sign == value:
                satisfied = True
                break # no need to check other literals in the clause
        # increment the count if the clause is satisfied
        if satisfied:
            count += 1
    # return the count as the fitness value
    return count

# define a function to perform simulated annealing on an individual
def simulated_annealing(individual):
    # define some parameters for simulated annealing
    initial_temperature = 1000 # initial temperature for annealing process
    final_temperature = 0.001 # final temperature for stopping condition
    alpha = 0.99 # constant factor for geometric cooling schedule
    beta = 1.01 # constant factor for adaptive cooling schedule 
    max_iterations = 1000 # maximum number of iterations to run
    max_restarts = 10 # maximum number of restarts to perform

    # evaluate the initial fitness of the individual
    current_fitness = fitness(individual)

    # check if the individual satisfies all clauses (fitness equals number of clauses)
    if current_fitness == len(formula.clauses):
        return individual # return the individual as a solution

    # initialize the temperature and iteration counter
    temperature = initial_temperature
    iteration = 0

    # initialize a tabu list to store recently visited solutions 
    tabu_list = []
    tabu_size = formula.nv // 10 # set the size of the tabu list to 10% of the number of variables

    # initialize a restart counter 
    restart = 0
    # counter for the iteration when the best fitness value was found
    best_iteration = 0 

    # run simulated annealing until a solution is found or the stopping condition is met
    while temperature > final_temperature and iteration < max_iterations and restart < max_restarts:
        # choose a random bit or bits to flip as a neighbor of the current individual depending on the temperature and fitness 
        if temperature > initial_temperature / 2: 
            # high temperature: flip one bit randomly 
            index = random.randint(0, formula.nv - 1)
            neighbor = individual.copy() # make a copy of the current individual
            neighbor[index] = not neighbor[index] # invert the bit value
        elif current_fitness < len(formula.clauses) / 2:
            # low fitness: flip two bits randomly 
            index1 = random.randint(0, formula.nv - 1)
            index2 = random.randint(0, formula.nv - 1)
            while index2 == index1: # avoid choosing the same bit twice
                index2 = random.randint(0, formula.nv - 1)
            neighbor = individual.copy() # make a copy of the current individual
            neighbor[index1] = not neighbor[index1] # invert the first bit value
            neighbor[index2] = not neighbor[index2] # invert the second bit value
        else:
            # medium temperature and fitness: flip one bit that appears in the most unsatisfied clauses 
            unsatisfied = [] # list to store the indices of unsatisfied clauses
            for i, clause in enumerate(formula.clauses):
                # check if the clause is unsatisfied by the individual
                satisfied = False
                for literal in clause:
                    # get the index and sign of the literal
                    index = abs(literal) - 1 # zero-based indexing
                    sign = literal > 0 # positive or negative literal
                    # get the value of the corresponding variable in the individual
                    value = individual[index]
                    # check if the literal and the value match
                    if sign == value:
                        satisfied = True
                        break # no need to check other literals in the clause
                # add the index of the clause to the list if it is unsatisfied
                if not satisfied:
                    unsatisfied.append(i)
            frequency = [0] * formula.nv # list to store the frequency of each variable in the unsatisfied clauses
            for i in unsatisfied:
                # loop through the literals in the unsatisfied clause
                for literal in formula.clauses[i]:
                    # get the index of the literal
                    index = abs(literal) - 1 # zero-based indexing
                    # increment the frequency of the variable by one
                    frequency[index] += 1
            max_frequency = max(frequency) # get the maximum frequency value 
            max_indices = [i for i, f in enumerate(frequency) if f == max_frequency] # get the indices of variables with maximum frequency 
            index = random.choice(max_indices) # choose one of them randomly 
            neighbor = individual.copy() # make a copy of the current individual
            neighbor[index] = not neighbor[index] # invert the bit value

        # evaluate the fitness of the neighbor
        neighbor_fitness = fitness(neighbor)

        # check if the neighbor satisfies all clauses (fitness equals number of clauses)
        if neighbor_fitness == len(formula.clauses):
            return neighbor # return the neighbor as a solution

        # calculate the change in fitness between the current individual and the neighbor
        delta_fitness = neighbor_fitness - current_fitness

        # check if the neighbor is better than or equal to the current individual or if it is accepted with some probability based on temperature and change in fitness
        if delta_fitness >= 0 or random.random() < math.exp(delta_fitness / temperature):
            # check if the neighbor is not in the tabu list 
            if tuple(neighbor) not in tabu_list:
                # update the current individual and fitness with the neighbor and its fitness
                individual = neighbor
                current_fitness = neighbor_fitness

        # add the current individual to the tabu list 
        tabu_list.append(tuple(individual))
        # remove the oldest individual from the tabu list if it exceeds its size 
        if len(tabu_list) > tabu_size:
            tabu_list.pop(0)

        # decrease or increase the temperature depending on whether there was an improvement or not 
        if delta_fitness > 0:
            # improvement: decrease temperature by multiplying it with alpha 
            temperature *= alpha
            # update the best iteration counter
            best_iteration = iteration
        else:
            # no improvement: increase temperature by multiplying it with beta 
            temperature *= beta 

        # increment the iteration counter
        iteration += 1

        # check if there was no improvement for a long time 
        if iteration - best_iteration > formula.nv:
            # perform a restart by reinitializing a random individual 
            individual = []
            for i in range(formula.nv):
                individual.append(random.choice([True, False])) # random bit value
            # reset the temperature and iteration counter 
            temperature = initial_temperature 
            iteration = 0 
            # increment the restart counter 
            restart += 1

    # return the final individual after simulated annealing process
    return individual

# initialize a random individual for simulated annealing
# initialize a random individual for simulated annealing
individual = []
for i in range(formula.nv):
    individual.append(random.choice([True, False])) # random bit value

# run simulated annealing on the individual and get the result 
result = simulated_annealing(individual)

# print the result of simulated annealing 
print("Result of simulated annealing:")
print(result)
