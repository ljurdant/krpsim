import threading
from loading import ft_progress
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
        init_population=None,
        min_dna_length=None,
        max_dna_length=None,
    ):
        self.population_size = population_size
        self.population = []
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.selection_rate = selection_rate
        self.generations = generations
        self.fitness_function = fitness_function
        self.init_population = init_population
        self.elite_rate = elite_rate
        self.min_dna_length = min_dna_length
        self.max_dna_length = max_dna_length
        self.genes = genes if genes is not None else []

    def sort_population(self):
        self.population.sort(key=lambda x: self.fitness_function(x), reverse=True)

    def parent_selection(self):
        """Select the best individuals from the population."""
        selection_point = int(self.population_size * self.selection_rate)

        return self.population[:selection_point]

    def elite_selection(self):
        """Select the best individuals from the population."""
        elite_count = int(self.population_size * self.elite_rate)
        return self.population[:elite_count]

    def crossover(self, individual1, individual2):
        """Perform crossover between two selected individuals and return two children."""
        child = []
        # copy_individual1 = copy.deepcopy(individual1)
        # copy_individual2 = copy.deepcopy(individual2)
        # if random.random() < self.crossover_rate:
        #     for _ in range(int(max(len(individual1), len(individual2)) / 2)):
        #         if len(copy_individual1) != 0:
        #             gene = random.choice(copy_individual1)
        #             child.append(gene)
        #             del copy_individual1[copy_individual1.index(gene)]
        #         if len(copy_individual2) != 0:
        #             gene = random.choice(copy_individual2)
        #             child.append(gene)
        #             del copy_individual2[copy_individual2.index(gene)]
        #     return child, True
        if random.random() < self.crossover_rate:
            crossover_point = min(
                random.randint(1, max(len(individual1), len(individual2)) - 1),
                min(len(individual1), len(individual2)) - 1,
            )
            child = individual1[:crossover_point] + individual2[crossover_point:]
            return child, True
        # No crossover: just clone the parents
        return individual1[:], False

    def crossover_generation(self, population):
        """Create a new generation by crossover."""
        crossover_population = []

        target_size = self.population_size - int(self.population_size * self.elite_rate)
        while len(crossover_population) < target_size:
            parent1 = random.choice(population)
            parent2 = random.choice(population)

            child, is_child = self.crossover(parent1, parent2)

            crossover_population.append(self.mutate(child))

        return crossover_population

    def mutate(self, individual):
        """Mutate an individual by randomly changing its genes."""
        mutated = copy.deepcopy(individual)
        for i in range(len(mutated)):
            if random.random() < self.mutation_rate:
                mutated[i] = random.choice(self.genes)
        return mutated

    def run(self):
        """Run the genetic algorithm."""
        self.population = self.init_population(self.population_size)
        fitnesses = []

        for _ in ft_progress(range(self.generations)):

            self.sort_population()
            fitnesses.append(self.fitness_function(self.population[0]))

            parent_population = self.parent_selection()
            elite_population = self.elite_selection()

            crossover_population = self.crossover_generation(parent_population)

            self.population = crossover_population + elite_population

            # cleanup population
        self.sort_population()
        fitnesses.append(self.fitness_function(self.population[0]))
        return self.population[0], fitnesses  # Return the best individual
