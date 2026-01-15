import streamlit as st
import random
import numpy as np

POP_SIZE = 300
CHROM_LENGTH = 80
GENERATIONS = 50
TARGET_ONES = 40
MAX_FITNESS = 80
MUTATION_RATE = 0.01
ELITISM_RATE = 0.05

def fitness(chromosome):
    ones = sum(chromosome)
    return MAX_FITNESS - abs(ones - TARGET_ONES)

def init_population():
    return [np.random.randint(0, 2, CHROM_LENGTH).tolist()
            for _ in range(POP_SIZE)]

# Selection (elitism)
def selection(population, fitnesses):
    elite_size = int(ELITISM_RATE * POP_SIZE)
    elite_indices = np.argsort(fitnesses)[-elite_size:]
    return [population[i] for i in elite_indices]

# Crossover
def crossover(parent1, parent2):
    point = random.randint(1, CHROM_LENGTH - 1)
    return parent1[:point] + parent2[point:]

# Mutation
def mutate(chromosome):
    for i in range(CHROM_LENGTH):
        if random.random() < MUTATION_RATE:
            chromosome[i] = 1 - chromosome[i]
    return chromosome

st.title("Genetic Algorithm: Bit Pattern Generator")

if st.button("Run Genetic Algorithm"):
    population = init_population()
    best_fitness_per_gen = []

    for gen in range(GENERATIONS):
        fitnesses = [fitness(ind) for ind in population]
        best_fitness_per_gen.append(max(fitnesses))

        elites = selection(population, fitnesses)
        new_population = elites.copy()

        while len(new_population) < POP_SIZE:
            parents = random.sample(elites, 2)
            child = crossover(parents[0], parents[1])
            child = mutate(child)
            new_population.append(child)

        population = new_population

    
    fitnesses = [fitness(ind) for ind in population]
    best_individual = population[np.argmax(fitnesses)]

    st.subheader("Best Bit Pattern Found")
    st.text("".join(map(str, best_individual)))
    st.write("Number of ones:", sum(best_individual))
    st.write("Fitness:", max(fitnesses))

    st.subheader("Fitness Progression")
    st.line_chart(best_fitness_per_gen)
