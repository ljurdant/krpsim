import multiprocessing

import random
import copy


class GeneticAlgorithm:
    def __init__(
        self,
        population_size=100,
        mutation_rate=0.01,
        crossover_rate=0.7,
        selection_rate=0.5,
        elite_rate=0.0,
        generations=100,
        genes=None,
        fitness_function=None,
    ):
        self.population_size = population_size
        self.population = []
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.selection_rate = selection_rate
        self.generations = generations
        self.fitness_function = fitness_function
        self.elite_rate = elite_rate
        self.genes = genes if genes is not None else []

    def init_population(self, min_dna_length, max_dna_length):
        """Initialize the population with random genes."""
        # Create a random population
        for _ in range(self.population_size):
            dna_length = random.randint(min_dna_length, max_dna_length)
            individual = [random.choice(self.genes) for _ in range(dna_length)]
            self.population.append(individual)

    def sort_population(self):
        # Perform parallel fitness evaluation
        fitness_scores = self.run_parallel_evaluation(self.population)

        # Sort population by fitness
        self.population = [
            pop_fit[0]
            for pop_fit in sorted(
                zip(self.population, fitness_scores),
                key=lambda x: x[1],
                reverse=True,
            )
        ]

    def parent_selection(self):
        """Select the best individuals from the population."""
        selection_point = int(self.population_size * self.selection_rate)
        elite_count = int(self.population_size * self.elite_rate)

        return self.population[:selection_point]

    def elite_selection(self):
        """Select the best individuals from the population."""
        elite_count = int(self.population_size * self.elite_rate)
        return self.population[:elite_count]

    def crossover(self, individual1, individual2):
        """Perform crossover between two selected individuals and return two children."""
        if random.random() < self.crossover_rate:
            crossover_point = random.randint(1, len(individual1) - 1)
            child1 = individual1[:crossover_point] + individual2[crossover_point:]
            child2 = individual2[:crossover_point] + individual1[crossover_point:]
            return child1, child2
        # No crossover: just clone the parents
        return individual1[:], individual2[:]

    def crossover_generation(self, population):
        """Create a new generation by crossover."""
        crossover_population = []

        target_size = self.population_size - int(self.population_size * self.elite_rate)
        while len(crossover_population) < target_size:
            parent1 = random.choice(population)
            parent2 = random.choice(population)

            child1, child2 = self.crossover(parent1, parent2)

            crossover_population.append(self.mutate(child1))
            if len(crossover_population) < target_size:
                crossover_population.append(self.mutate(child2))

        return crossover_population

    def mutate(self, individual):
        """Mutate an individual by randomly changing its genes."""
        mutated = copy.deepcopy(individual)
        for i in range(len(mutated)):
            if random.random() < self.mutation_rate:
                mutated[i] = random.choice(self.genes)
        return mutated

    def parallel_fitness(self, individual):
        """Parallel fitness evaluation for an individual."""
        return self.fitness_function(individual)

    def run_parallel_evaluation(self, population):
        """Evaluate the fitness of the population in parallel."""
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            fitness_scores = pool.map(self.parallel_fitness, population)
        return fitness_scores

    def run(self, min_dna_length, max_dna_length):
        """Run the genetic algorithm."""
        self.init_population(min_dna_length, max_dna_length)

        for _ in range(self.generations):
            self.sort_population()

            parent_population = self.parent_selection()
            elite_population = self.elite_selection()

            crossover_population = self.crossover_generation(parent_population)

            self.population = crossover_population + elite_population
        self.sort_population()
        return self.population[0]  # Return the best individual
