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
        fitness_sharing=None,
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
        self.fitness_sharing = fitness_sharing
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
        crossover_point = min(
            random.randint(1, max(len(individual1), len(individual2)) - 1),
            min(len(individual1), len(individual2)) - 1,
        )
        child = individual1[:crossover_point] + individual2[crossover_point:]
        return child

    def tournament_selection(self, population, k=2):
        """
        Perform a tournament of size k on the population.
        Return the best individual (highest fitness) among k random picks.
        """
        # Randomly choose k individuals
        tournament_contestants = random.sample(population, k)

        # Identify best
        return max(tournament_contestants, key=lambda x: self.fitness_function(x))

    def create_offspring(self, population, k=2):
        """
        Example function to create one child using tournament selection and crossover.
        """
        # 1) Select parents via tournament
        parent1 = self.tournament_selection(population, k)
        parent2 = self.tournament_selection(population, k)

        if random.random() < self.crossover_rate:
            child = self.crossover(parent1, parent2)
        else:
            child = random.choice([parent1[:], parent2[:]])
        return child

    def crossover_generation(self, population):
        """Create a new generation by crossover."""
        crossover_population = []

        target_size = self.population_size - int(self.population_size * self.elite_rate)
        while len(crossover_population) < target_size:

            child = self.create_offspring(population, 3)

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

        print(sorted([len(pop) for pop in self.population]))
        for _ in ft_progress(range(self.generations)):

            raw_fitnesses = [self.fitness_function(ind) for ind in self.population]

            shared_fitnesses = self.fitness_sharing(
                self.population,
                raw_fitnesses,
            )

            # 4) Selection using shared_fitnesses
            # Example: Sort and pick top half
            zipped = list(zip(self.population, shared_fitnesses))
            zipped.sort(key=lambda x: x[1], reverse=True)
            # keep top half
            half = int(self.population_size * self.selection_rate)
            parents = [zipped[i][0] for i in range(half)]

            self.sort_population()
            # print each individual fitness
            for ind in self.population:
                print(
                    int(self.fitness_function(ind) * 1000),
                    end=", ",
                )
            print()
            fitnesses.append(self.fitness_function(self.population[0]))

            # parent_population = self.parent_selection()
            elite_population = self.elite_selection()

            crossover_population = self.crossover_generation(parents)

            self.population = crossover_population + elite_population

            # cleanup population
        self.sort_population()
        fitnesses.append(self.fitness_function(self.population[0]))
        return self.population[0], fitnesses  # Return the best individual
