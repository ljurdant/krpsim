#!/usr/bin/env python3

from _types import Process
from typing import List
from genetic_algorithm import GeneticAlgorithm
import random
from parser import parse
import sys
import json


from datetime import datetime


from matplotlib import pyplot as plt


def can_run_task(stock: dict[str, int], needs: dict[str, int]) -> bool:
    """
    Check if a task can be run with the current stock.
    """
    for need, amount in needs.items():
        if stock.get(need, 0) < amount:
            return False
    return True


def do_process(
    process_name: str, processes: dict[str, Process], stock: dict[str, int]
) -> int:
    """Do a process and update the stock."""
    # Check if we have enough resources
    process = processes[process_name]

    for resource, amount in process["need"].items():
        if stock.get(resource, 0) < amount:
            return 0, False

    # Update the stock
    for resource, amount in process["need"].items():
        stock[resource] -= amount

    # Add the result to the stock
    for resource, amount in process["result"].items():
        stock[resource] = stock.get(resource, 0) + amount

    return process["time"], True


def generate_feasible_individual(
    processes: dict[str, Process],
    initial_stock: dict[str, int],
    max_length=30,
) -> List[str]:
    """
    Returns a random feasible chromosome built via SGS,
    stopping when no more tasks are feasible or max_length is reached.
    """

    stock_copy = initial_stock.copy()
    chromosome = []
    task_names = list(processes.keys())  # Genes
    for _ in range(max_length):
        feasible_tasks = []
        for task_name in task_names:
            # Check if we can run 'task_name' with current stock_copy
            if can_run_task(stock_copy, processes[task_name]["need"]):
                feasible_tasks.append(task_name)

        if not feasible_tasks:
            # No more tasks can be run
            break

        # Randomly select one feasible task
        chosen_task = random.choice(feasible_tasks)

        # Append to chromosome
        chromosome.append(chosen_task)

        # Update stock
        do_process(chosen_task, processes, stock_copy)

    return chromosome


def generate_random_individual(
    processes: dict[str, Process],
    min_length: int = 0,
    max_length=30,
) -> List[str]:

    task_names = list(processes.keys())  # Genes
    length = random.randint(int(3 * max_length / 4), max_length)
    chromosome = [random.choice(task_names) for _ in range(length)]

    return chromosome


def get_resource_hierarchy(
    processes: dict[str, Process], optimize: List[str]
) -> dict[str, List[str]]:

    resources = set(
        [key for process in processes.values() for key in process["need"].keys()]
        + [key for process in processes.values() for key in process["result"].keys()]
    )

    hierarchy_dict = {}
    for opt in optimize:
        hierarchy_dict[opt] = 1

    while hierarchy_dict.keys() != resources:
        for process in [
            process
            for process in processes.values()
            if any(key in hierarchy_dict.keys() for key in process["result"].keys())
        ]:
            for need in process["need"].keys():
                for result in [
                    result
                    for result in process["result"].keys()
                    if result in hierarchy_dict.keys()
                ]:
                    need_value = hierarchy_dict[result] / process["need"][need]
                    if hierarchy_dict.get(need) is None:
                        hierarchy_dict[need] = need_value
                    else:
                        hierarchy_dict[need] = min(need_value, hierarchy_dict[need])

    for opt in optimize:
        hierarchy_dict[opt] = 2

    return hierarchy_dict


def get_score(
    individual: List[Process],
    processes,
    stock: dict[str, int],
    optimize: List[str],
    hierarchy: dict[str, float],
) -> float:

    total_time = 0
    valid_count = 0
    for process_name in individual:
        time_taken, success = do_process(process_name, processes, stock)
        total_time += time_taken
        valid_count += int(success)
        # if not success:
        # break

    resources_count = sum(
        ((stock.get(resource, 0) * hierarchy[resource])) for resource in stock.keys()
    )

    if "time" in optimize:
        return resources_count / total_time if total_time > 0 else 0
    else:
        return resources_count  # + valid_count / len(individual)
    # I need to add valid_count to the score or something to reward good ressources collected


def get_stock_after_individual(
    individual: List[Process], processes, stock: dict[str, int]
) -> dict[str, int]:

    for process_name in individual:
        do_process(process_name, processes, stock)

    return stock


def get_min_max_gene_length(
    min_length: int,
    max_length: int,
    processes: dict[str, Process],
    opt: str,
    stock: dict[str, int],
) -> tuple[int, int]:
    """Get the min and max gene length."""

    if opt == "time":
        return 0, 0
    for process in processes.values():
        if opt in process["result"]:
            for resource, amount in process["need"].items():
                if resource in stock:
                    min_length += 1
                    max_length *= amount
                else:
                    min_length, max_length = get_min_max_gene_length(
                        min_length + 1, max_length * amount, processes, resource, stock
                    )
            break

    return min_length, max_length


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python main.py <path_to_krpsim_config>")
        sys.exit(1)

    config_file = sys.argv[1]
    stock, processes, optimize = parse(config_file)

    _min = 0
    _max = 0
    for opt in optimize:
        tmp_min, tmp_max = get_min_max_gene_length(0, 1, processes, opt, stock)
        _min = _min + tmp_min
        _max = _max + tmp_max

    _max = min(_max, 1800)

    print("Min gene length:", _min)
    print("Max gene length:", _max)
    hierarchy = get_resource_hierarchy(processes, optimize)
    print("Hierarchy:", hierarchy)

    def fitness_function(individual):
        stock_copy = stock.copy()
        processes_copy = processes.copy()
        optimize_copy = optimize.copy()
        return get_score(
            individual, processes_copy, stock_copy, optimize_copy, hierarchy
        )

    def cleanup_individuals(individuals):
        for individual in individuals:
            copy_stock = stock.copy()
            i = 0
            j = 0
            while i < len(individual) and j < len(individual):
                while i < len(individual) and can_run_task(
                    copy_stock, processes[individual[i]]["need"]
                ):
                    do_process(individual[i], processes, copy_stock)
                    i += 1
                    j += 1
                if i >= len(individual):
                    break
                individual = individual[:i] + individual[i + 1 :] + [individual[i]]
                j += 1

        return individuals

    def init_population_with_sgs(pop_size):
        population = []
        rand_portion = 0.9
        rand_pop_size = int(pop_size * rand_portion)
        feasible_pop_size = pop_size - rand_pop_size

        for _ in range(feasible_pop_size):
            individual = generate_feasible_individual(processes, stock, _max)
            population.append(individual)

        for _ in range(rand_pop_size):
            individual = generate_random_individual(processes, _min, _max)
            population.append(individual)

        return population

    ga = GeneticAlgorithm(
        population_size=1000,
        crossover_rate=0.6,
        elite_rate=0.01,
        selection_rate=0.5,
        mutation_rate=0.02,
        genes=list(processes.keys()),
        fitness_function=fitness_function,
        init_population=init_population_with_sgs,
        generations=10,
        parent_selection_type="tournament",
        selection_pressure=2,
        tournament_probability=0.7,
        crossover_point="uniform",
        min_dna_length=_min,
        max_dna_length=_max,
        cleanup_individuals=cleanup_individuals,
        # New hypermutation parameters:
        hypermutation_factor=0.05,  # how many times to multiply the mutation rate
        hypermutation_trigger_gens=3,  # how many generations of no improvement before we trigger
        hypermutation_duration=2,  # how many generations to keep hypermutation on
    )

    best, fitnesses = ga.run()

    fitness = fitness_function(best)
    print("Best fitness:", fitness)

    print("Best individual:", best, len(best))
    valid_best = []
    copy = stock.copy()
    for process in best:
        _, success = do_process(process, processes, copy)
        if success:
            valid_best.append(process)
    print(
        "Stock after best individual:",
        get_stock_after_individual(best, processes, stock.copy()),
    )

    now = datetime.now()

    current_time = now.strftime("%Y%m%d_%H:%M:%S")
    print("Time:", current_time)

    json.dump(
        valid_best,
        open(
            f"../results/{config_file.split('/')[-1]}_{fitness*100:.2f}_{current_time}.json",
            "w",
        ),
        indent=4,
    )
    plt.plot(fitnesses)
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("Fitness over generations")
    plt.show()
