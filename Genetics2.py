import random
import numpy as np
from pysat.formula import CNF
from pysat.solvers import Solver

def read_cnf(file_path):
    formula = CNF(from_file=file_path)
    return formula

def fitness(formula, individual):
    satisfied_clauses = 0
    for clause in formula.clauses:
        for literal in clause:
            if literal > 0 and individual[literal - 1] or literal < 0 and not individual[-literal - 1]:
                satisfied_clauses += 1
                break
    return satisfied_clauses

def genetic_algorithm(formula, population_size=100, generations=100, mutation_rate=0.1):
    population = [np.random.choice([True, False], formula.nv) for _ in range(population_size)]

    for generation in range(generations):
        fitness_values = [fitness(formula, individual) for individual in population]
        if max(fitness_values) == len(formula.clauses):
            break

        new_population = []
        for _ in range(population_size):
            parents = random.choices(population, weights=fitness_values, k=2)
            crossover_point = random.randint(1, formula.nv - 1)
            offspring = np.concatenate((parents[0][:crossover_point], parents[1][crossover_point:]))

            if random.random() < mutation_rate:
                mutation_point = random.randint(0, formula.nv - 1)
                offspring[mutation_point] = not offspring[mutation_point]

            new_population.append(offspring)

        population = new_population

    best_individual = max(population, key=lambda x: fitness(formula, x))
    return best_individual


def main():
    formula = read_cnf("Input.cnf")

    ga_solution = genetic_algorithm(formula)

    print("Genetic Algorithm solution:", " ".join(str(i + 1) if v else str(-(i + 1)) for i, v in enumerate(ga_solution)))

if __name__ == "__main__":
    main()
