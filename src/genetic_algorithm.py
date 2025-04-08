import random


class GeneticAlgorithm:
    def __init__(
        self,
        population_size=100,
        mutation_rate=0.01,
        crossover_rate=0.7,
        generations=100,
        genes=None,
        fitness_function=None,
    ):
        self.population_size = population_size
        self.population = []
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.generations = generations
        self.fitness_function = fitness_function
        self.genes = genes if genes is not None else []

    def init_population(self, dna_length):
        """Initialize the population with random genes."""
        self.population = [
            [random.choice(self.genes) for _ in range(dna_length)]
            for _ in range(self.population_size)
        ]

    def selection(self):
        """Select the best individuals from the population."""
        self.population.sort(key=self.fitness_function, reverse=True)
        halfway_point = self.population_size // 2
        return self.population[:halfway_point]

    def crossover(self, individual1, individual2):
        """Perform crossover between two selected individuals."""
        if random.random() < self.crossover_rate:
            crossover_point = random.randint(1, len(individual1) - 1)
            return individual1[:crossover_point] + individual2[crossover_point:]

        return individual1

    def crossover_generation(self, population):
        """Create a new generation by crossover."""
        crossover_population = []
        halfway_point = self.population_size // 2
        for _ in range(self.population_size - halfway_point):
            parent1 = random.choice(population)
            parent2 = random.choice(population)

            child1 = self.crossover(parent1, parent2)

            crossover_population.append(self.mutate(child1))
        return crossover_population

    def mutate(self, individual):
        """Mutate an individual by randomly changing its genes."""
        for i in range(len(individual)):
            if random.random() < self.mutation_rate:
                individual[i] = random.choice(self.genes)
        return individual

    def run(self, dna_length):
        """Run the genetic algorithm."""
        self.init_population(dna_length)

        for _ in range(self.generations):
            good_population = self.selection()

            crossover_population = self.crossover_generation(good_population)
            mutated_population = [
                self.mutate(individual) for individual in crossover_population
            ]
            self.population = mutated_population + good_population

        return self.selection()[0]
