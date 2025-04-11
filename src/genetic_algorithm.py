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
        selection_pressure=2,
        tournament_probability=0.9,
        parent_selection_type="random",
        crossover_point="single",
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
        self.selection_pressure = selection_pressure
        self.tournament_probability = tournament_probability
        self.parent_selection_type = parent_selection_type
        self.crossover_point = crossover_point
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

    def single_point_crossover(self, individual1, individual2):
        crossover_point = min(
            random.randint(1, max(len(individual1), len(individual2)) - 1),
            min(len(individual1), len(individual2)) - 1,
        )
        return individual1[:crossover_point] + individual2[crossover_point:]

    def uniform_crossover(self, individual1, individual2):
        """Perform uniform crossover between two selected individuals and return two children."""
        child = []
        length = len(individual1)
        if len(individual2) != len(individual1):
            length = random.randint(
                min(len(individual1), len(individual2)),
                max(len(individual1), len(individual2)),
            )
        for i in range(length):
            if i > len(individual1) - 1:
                child.append(individual2[i])
                continue
            if i > len(individual2) - 1:
                child.append(individual1[i])
                continue
            child.append(random.choice([individual1[i], individual2[i]]))

        return child

    def crossover(self, individual1, individual2):
        """Perform crossover between two selected individuals and return two children."""
        if self.crossover_point == "single":
            return self.single_point_crossover(individual1, individual2)
        elif self.crossover_point == "uniform":
            return self.uniform_crossover(individual1, individual2)
        else:
            raise ValueError("Invalid crossover point type.")

    def get_crossover_parent(self, population):
        """
        Select two parents for crossover using tournament selection.
        """
        if self.parent_selection_type == "tournament":
            return self.tournament_selection(population)
        else:
            return random.choice(population)

    def tournament_selection(self, population):
        """
        Perform a tournament of size k on the population.
        Return the best individual (highest fitness) among k random picks.
        """

        p = self.tournament_probability
        # Randomly choose k individuals
        tournament_contestants = random.sample(population, self.selection_pressure)
        tournament_contestants.sort(
            key=lambda x: self.fitness_function(x), reverse=True
        )
        weights = [p * ((1 - p) ** i) for i in range(len(tournament_contestants))]
        return random.choices(population=tournament_contestants, k=1, weights=weights)[
            0
        ]
        # Identify best

    def create_offspring(self, population):
        """
        Example function to create one child using tournament selection and crossover.
        """
        # 1) Select parents via tournament

        parent1 = self.get_crossover_parent(population)
        parent2 = self.get_crossover_parent(population)

        if random.random() < self.crossover_rate:
            child = self.crossover(parent1, parent2)
        else:
            child = max(
                [parent1[:], parent2[:]], key=lambda x: self.fitness_function(x)
            )
        return child

    def crossover_generation(self, population):
        """Create a new generation by crossover."""
        crossover_population = []

        target_size = self.population_size - int(self.population_size * self.elite_rate)
        while len(crossover_population) < target_size:

            child = self.create_offspring(population)

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
        best = max(self.population, key=lambda x: self.fitness_function(x))
        fitnesses.append(self.fitness_function(best))
        return (
            best,
            fitnesses,
        )  # Return the best individual
