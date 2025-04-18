from loading import ft_progress
import random
import time
import os
import copy
from functools import cmp_to_key


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
        time_limit=None,
        init_population=None,
        hyper_mutation_rate=0.01,
        hyper_tournament_probability=0.7,
        hyper_selection_pressure=2,
        hyper_change_frequency=100,
        hyper_numb_generation=3,
        max_gene_length=1,
        min_gene_length=1,
        cmp_function=None,
        get_extra_params=None,
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
        self.genes = genes if genes is not None else []
        self.time_limit = time_limit
        self.best_individual = None
        self.stagnation_count = 0
        self.hyper_mutation_count = 0
        self.hyper_mutation_rate = hyper_mutation_rate
        self.hyper_change_frequency = hyper_change_frequency
        self.hyper_tournament_probability = hyper_tournament_probability
        self.hyper_selection_pressure = hyper_selection_pressure
        self.hyper_numb_generation = hyper_numb_generation
        self.max_gene_length = max_gene_length
        self.min_gene_length = min_gene_length
        self.cmp_function = cmp_function
        if cmp_function is None:
            self.cmp_function = lambda x, y: (
                self.fitness_function(x) - self.fitness_function(y)
            )
        self.get_extra_params = get_extra_params

    def sort_population(self):

        self.population.sort(key=cmp_to_key(self.cmp_function), reverse=True)

    def parent_selection(self):
        """Select the best individuals from the population."""
        selection_point = int(self.population_size * self.selection_rate)

        return self.population[:selection_point]

    def elite_selection(self):
        """Select the best individuals from the population."""
        elite_count = int(self.population_size * self.elite_rate)
        return self.population[:elite_count]

    def single_point_crossover(self, individual1, individual2):
        dna_length1 = sum(gene["amount"] for gene in individual1)
        dna_length2 = sum(gene["amount"] for gene in individual2)
        # crossover_point = random.randint(0, dna_length - 1)
        crossover_point = random.randint(
            0,
            random.randint(
                min(dna_length1 - 1, dna_length2 - 1),
                max(dna_length1 - 1, dna_length2 - 1),
            ),
        )
        child = []

        total_length = 0
        i = 0
        while total_length < crossover_point and i < len(individual1):
            if total_length + individual1[i]["amount"] > crossover_point:
                amount1 = crossover_point - total_length
                if len(child) and individual1[i]["process"] == child[-1]["process"]:
                    child[-1]["amount"] += amount1
                else:
                    child.append(
                        {
                            "process": individual1[i]["process"],
                            "amount": amount1,
                            "parallel": individual1[i]["parallel"],
                        }
                    )

                break
            else:
                child.append(
                    {
                        "process": individual1[i]["process"],
                        "amount": individual1[i]["amount"],
                        "parallel": individual1[i]["parallel"],
                    }
                )
                total_length += individual1[i]["amount"]
                i += 1

        total_length = 0
        i = 0
        while total_length < crossover_point and i < len(individual2):
            if total_length + individual2[i]["amount"] > crossover_point:
                amount2 = total_length + individual2[i]["amount"] - crossover_point
                if len(child) and individual2[i]["process"] == child[-1]["process"]:
                    child[-1]["amount"] += amount2
                else:
                    child.append(
                        {
                            "process": individual2[i]["process"],
                            "amount": amount2,
                            "parallel": individual2[i]["parallel"],
                        }
                    )
                i += 1
                break
            else:
                total_length += individual2[i]["amount"]
                i += 1

        if i < len(individual2):
            for gene in individual2[i:]:
                if len(child) and gene["process"] == child[-1]["process"]:
                    child[-1]["amount"] += gene["amount"]
                else:
                    child.append(
                        {
                            "process": gene["process"],
                            "amount": gene["amount"],
                            "parallel": gene["parallel"],
                        }
                    )

        return child

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
        tournament_probability = self.tournament_probability
        selection_pressure = self.selection_pressure
        if self.stagnation_count >= self.hyper_change_frequency:
            selection_pressure = random.randint(
                min(self.selection_pressure, self.hyper_selection_pressure),
                max(self.selection_pressure, self.hyper_selection_pressure),
            )
        if self.stagnation_count >= self.hyper_change_frequency:
            tournament_probability = self.hyper_tournament_probability
        p = tournament_probability
        # Randomly choose k individuals
        tournament_contestants = random.sample(population, selection_pressure)
        tournament_contestants.sort(key=cmp_to_key(self.cmp_function), reverse=True)
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
        crossover_rate = self.crossover_rate

        if random.random() < crossover_rate:
            child_individual = self.crossover(parent1[0], parent2[0])
            child = (
                child_individual,
                self.get_extra_params(child_individual),
            )
        else:
            child = max(parent1, parent2, key=cmp_to_key(self.cmp_function))
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
        mutated_individual = []
        mutated = False
        mutation_rate = self.mutation_rate
        if self.stagnation_count >= self.hyper_change_frequency:
            mutation_rate = self.hyper_mutation_rate
        for gene in individual[0]:
            mutated_individual.append(
                {
                    "process": gene["process"],
                    "amount": gene["amount"],
                    "parallel": gene["parallel"],
                }
            )
            if random.random() < mutation_rate:
                mutated = True
                mutated_individual[-1]["process"] = random.choice(self.genes)

            if random.random() < mutation_rate:
                mutated = True
                mutated_individual[-1]["amount"] = random.randint(
                    self.min_gene_length, self.max_gene_length
                )
            if random.random() < mutation_rate:
                mutated = True
                mutated_individual[-1]["parallel"] = not mutated_individual[-1][
                    "parallel"
                ]

        if mutated:
            return mutated_individual, self.get_extra_params(mutated_individual)
        else:
            return individual

    def one_generation(self):
        self.sort_population()

        if self.best_individual is not None and self.fitness_function(
            self.population[0]
        ) == self.fitness_function(self.best_individual):
            self.stagnation_count += 1
        else:
            self.stagnation_count = 0
        if self.stagnation_count >= self.hyper_change_frequency:
            self.hyper_mutation_count += 1
        if self.hyper_mutation_count >= self.hyper_numb_generation:
            self.hyper_mutation_count = 0
            self.stagnation_count = 0

        parent_population = self.parent_selection()
        elite_population = self.elite_selection()

        self.best_individual = self.population[0]
        self.fitnesses.append(self.fitness_function(self.best_individual))

        crossover_population = self.crossover_generation(parent_population)
        self.population = crossover_population + elite_population

    def run(self):
        """Run the genetic algorithm."""
        self.population = self.init_population(self.population_size)
        self.fitnesses = []

        if self.time_limit:
            start = time.time()
            end = start + self.time_limit
            generations = 0
            message = ""
            width = os.get_terminal_size()[0]
            while time.time() < end:
                self.one_generation()
                if generations >= 1:
                    moveup = "\033[A"
                    print(moveup * int(len(message) / width + 1))
                fitness = self.fitness_function(self.best_individual)
                message = f"ETA {end - time.time():2.0f}s : Generation {generations:4.0f} | Best fitness {fitness:3.2f} "
                print(message, end="\r")
                generations += 1
            print()
            print(f"Time limit reached after {generations} generations")
        else:
            for _ in ft_progress(range(self.generations)):
                self.one_generation()
        best = max(self.population, key=cmp_to_key(self.cmp_function))

        self.fitnesses.append(self.fitness_function(best))
        return (
            best,
            self.fitnesses,
        )  # Return the best individual
