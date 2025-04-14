from loading import ft_progress
import random


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
        cleanup_individuals=None,
        genes=None,
        fitness_function=None,
        init_population=None,
        min_dna_length=None,
        max_dna_length=None,
        # New hypermutation parameters:
        hypermutation_factor=3.0,  # how many times to multiply the mutation rate
        hypermutation_trigger_gens=5,  # how many generations of no improvement before we trigger
        hypermutation_duration=2,  # how many generations to keep hypermutation on
        random_immigrant_rate=0.05,
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
        self.cleanup_individuals = cleanup_individuals
        # ... existing init ...
        self.hypermutation_factor = hypermutation_factor
        self.hypermutation_trigger_gens = hypermutation_trigger_gens
        self.hypermutation_duration = hypermutation_duration

        # Track improvement
        self.generations_since_improvement = 0
        self.best_fitness_so_far = None
        self.hypermutation_active = False
        self.hypermutation_timer = 0

        self.random_immigrant_rate = random_immigrant_rate

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
        # If hypermutation is active, multiply the base mutation rate
        effective_mutation_rate = self.mutation_rate
        if self.hypermutation_active:
            effective_mutation_rate *= self.hypermutation_factor

        for i in range(len(individual)):
            if random.random() < effective_mutation_rate:
                individual[i] = random.choice(self.genes)
        return individual

    def _random_individual(self):
        # your code that returns a random chromosome
        # for example:
        length = random.randint(self.min_dna_length, self.max_dna_length)
        return [random.choice(self.genes) for _ in range(length)]

    def run(self):
        """Run the genetic algorithm."""
        self.population = self.init_population(self.population_size)
        fitnesses = []

        for _ in ft_progress(range(self.generations)):
            self.sort_population()

            current_best_fitness = self.fitness_function(self.population[0])

            # 2) Check for improvement
            if (
                self.best_fitness_so_far is None
                or current_best_fitness > self.best_fitness_so_far
            ):
                self.best_fitness_so_far = current_best_fitness
                self.generations_since_improvement = 0
                if self.hypermutation_active:
                    # Turn off hypermutation
                    print(
                        f"Hypermutation / hyperselection_pressure OFF at generation {_}"
                    )
                    self.hypermutation_active = False
                    self.selection_pressure = 2
            else:
                self.generations_since_improvement += 1

            # 3) Possibly trigger hypermutation
            if not self.hypermutation_active:
                if (
                    self.generations_since_improvement
                    >= self.hypermutation_trigger_gens
                ):
                    # Turn on hypermutation
                    self.hypermutation_active = True
                    self.selection_pressure = 8
                    # self.hypermutation_timer = self.hypermutation_duration
                    print(
                        f"Hypermutation / hyperselection_pressure ON at generation {_}"
                    )
            fitnesses.append(current_best_fitness)

            parent_population = self.parent_selection()
            elite_population = self.elite_selection()

            crossover_population = self.crossover_generation(parent_population)
            # crossover_population = self.cleanup_individuals(crossover_population)
            new_generation = crossover_population + elite_population

            if self.hypermutation_active:
                immigrant_count = int(self.population_size * self.random_immigrant_rate)
                for _ in range(immigrant_count):
                    rand_idx = random.randint(0, len(new_generation) - 1)
                    # create_random_individual or some feasible generator
                    new_generation[rand_idx] = self._random_individual()
            self.population = new_generation
            # cleanup population
        best = max(self.population, key=lambda x: self.fitness_function(x))
        fitnesses.append(self.fitness_function(best))
        return (
            best,
            fitnesses,
        )  # Return the best individual
