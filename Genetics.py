import random
import pysat
from pysat.formula import CNF
from pysat.solvers import Solver


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

# define a function to simplify the formula using unit propagation
def simplify(formula):
    # create a copy of the formula
    simplified_formula = CNF()
    simplified_formula.clauses = formula.clauses.copy()
    simplified_formula.nv = formula.nv
    # create a list to store the unit clauses
    unit_clauses = []
    # loop until no more unit clauses are found
    while True:
        # find all unit clauses in the formula
        for clause in simplified_formula.clauses:
            if len(clause) == 1:
                # add the unit clause to the list if it is not already there
                if clause[0] not in unit_clauses and -clause[0] not in unit_clauses:
                    unit_clauses.append(clause[0])
        # check if any unit clauses are found
        if len(unit_clauses) == 0:
            # no more unit clauses: exit the loop
            break
        # propagate the unit clauses in the formula
        for literal in unit_clauses:
            # remove all clauses that contain the literal
            simplified_formula.clauses = [c for c in simplified_formula.clauses if literal not in c]
            # remove the negation of the literal from all clauses that contain it
            simplified_formula.clauses = [list(filter(lambda x: x != -literal, c)) for c in simplified_formula.clauses]
            # update the number of variables in the formula
            simplified_formula.nv -= 1
    # return the simplified formula and the unit clauses
    return simplified_formula, unit_clauses

# define a function to evaluate the fitness of an individual
def fitness(individual, formula):
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

# define a function to perform crossover between two individuals
def crossover(parent1, parent2):
    # choose a random point to cut and swap the parts with some probability
    point = random.randint(0, formula.nv - 1)
    child1 = []
    child2 = []
    for i in range(formula.nv):
        # use uniform crossover with probability 0.5
        if random.random() < 0.5:
            child1.append(parent1[i])
            child2.append(parent2[i])
        # otherwise use one-point crossover
        else:
            if i < point:
                child1.append(parent1[i])
                child2.append(parent2[i])
            else:
                child1.append(parent2[i])
                child2.append(parent1[i])
    # return the two children as a tuple
    return (child1, child2)

# define a function to perform mutation on an individual
def mutation(individual, formula):
    # choose a random bit to flip only if it improves the fitness of the individual
    index = random.randint(0, formula.nv - 1)
    original_fitness = fitness(individual, formula) # get the original fitness of the individual
    individual[index] = not individual[index] # invert the bit value
    new_fitness = fitness(individual, formula) # get the new fitness of the individual
    # check if the new fitness is better than or equal to the original fitness
    if new_fitness >= original_fitness:
        # keep the mutation
        return
    else:
        # undo the mutation
        individual[index] = not individual[index]
        return

# define some parameters for the genetic algorithm
population_size = 100 # number of individuals in the population
crossover_rate = 0.8 # probability of performing crossover
mutation_rate = 0.1 # probability of performing mutation
max_generations = 1000 # maximum number of generations to run

# simplify the formula using unit propagation
simplified_formula, unit_clauses = simplify(formula)

# initialize the population with individuals that are more likely to satisfy the formula using a local search algorithm 
population = []
for i in range(population_size):
    individual = []
    for j in range(simplified_formula.nv):
        individual.append(random.choice([True, False])) # random bit value
    # perform local search on the individual using a solver
    solver = Solver()
    solver.append_formula(simplified_formula.clauses)
    solver.solve(assumptions = [i + 1 if individual[i] else -(i + 1) for i in range(simplified_formula.nv)] ) # use the individual as an initial assignment 
    model = solver.get_model() # get the model returned by the solver 
    if model: 
        # if the model is not None, use it as the individual 
        individual = [abs(literal) > 0 for literal in model]
    population.append(individual)

# run the genetic algorithm until a solution is found or the maximum number of generations is reached
found = False # flag to indicate if a solution is found
generation = 0 # counter for the current generation

while not found and generation < max_generations:
    # evaluate the fitness of each individual in the population
    fitness_values = []
    for individual in population:
        fitness_values.append(fitness(individual, simplified_formula))

    # check if any individual satisfies all clauses (fitness equals number of clauses)
    # check if any individual satisfies all clauses (fitness equals number of clauses)
    best_fitness = max(fitness_values) # get the best fitness value in the population
    best_index = fitness_values.index(best_fitness) # get the index of the best individual
    best_individual = population[best_index] # get the best individual
    if best_fitness == len(simplified_formula.clauses):
        found = True # set the flag to true
        break # exit the loop

    # select individuals for the next generation based on their fitness values
    # using roulette wheel selection
    total_fitness = sum(fitness_values) # get the total fitness of the population
    probabilities = [f / total_fitness for f in fitness_values] # get the normalized probabilities for each individual
    new_population = [] # list to store the new population
    for i in range(population_size):
        # choose a random number between 0 and 1
        r = random.random()
        # find the first individual whose cumulative probability is greater than or equal to r
        cumulative = 0
        for j in range(population_size):
            cumulative += probabilities[j]
            if cumulative >= r:
                # add that individual to the new population
                new_population.append(population[j])
                break

    # perform crossover on pairs of individuals in the new population
    for i in range(0, population_size, 2):
        # check if crossover should be performed
        if random.random() < crossover_rate:
            # get the two parents
            parent1 = new_population[i]
            parent2 = new_population[i+1]
            # perform crossover and get the two children
            child1, child2 = crossover(parent1, parent2)
            # replace the parents with the children in the new population
            new_population[i] = child1
            new_population[i+1] = child2

    # perform mutation on each individual in the new population
    for i in range(population_size):
        # check if mutation should be performed
        if random.random() < mutation_rate:
            # get the individual
            individual = new_population[i]
            # perform mutation on the individual
            mutation(individual, simplified_formula)
            # replace the individual in the new population
            new_population[i] = individual

    # update the population with the new population
    population = new_population

    # increment the generation counter
    generation += 1

# print the result of the genetic algorithm
if found:
    print("Solution found by genetic algorithm:")
    print(best_individual + unit_clauses) # append the unit clauses to the solution 
else:
    print("No solution found by genetic algorithm after {} generations.".format(generation))
