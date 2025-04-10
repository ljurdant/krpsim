#!/usr/bin/env python3

from _types import Process
from typing import List
from genetic_algorithm import GeneticAlgorithm
import random
from parser import parse
import sys
from collections import Counter
import time

from matplotlib import pyplot as plt


def can_run_task(stock: dict[str, int], needs: dict[str, int]) -> bool:
    """
    Check if a task can be run with the current stock.
    """
    for need, amount in needs.items():
        if stock.get(need, 0) < amount:
            return False
    return True


def run_task(stock: dict[str, int], task: dict[str, int]) -> None:
    """
    Run a task and update the stock.
    """
    for resource, amount in task["need"].items():
        stock[resource] = stock.get(resource, 0) - amount

    # Add the result to the stock
    for resource, amount in task["result"].items():
        stock[resource] = stock.get(resource, 0) + amount


def generate_feasible_individual(processes, initial_stock, max_length=30):
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
        run_task(stock_copy, processes[chosen_task])

    return chromosome


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
        #     break

    resources_count = sum(
        ((stock.get(resource, 0) * hierarchy[resource])) for resource in stock.keys()
    )

    if "time" in optimize:
        return resources_count / total_time if total_time > 0 else 0
    else:
        return resources_count + valid_count / len(individual)
    # I need to add valid_count to the score or something to reward good ressources collected


def get_stock_after_individual(
    individual: List[Process], processes, stock: dict[str, int]
) -> dict[str, int]:

    for process_name in individual:
        do_process(process_name, processes, stock)

    return stock


def measure_time(func):
    """Decorator that measures the execution time of the decorated function."""

    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        print(f"{func.__name__} executed in {end_time - start_time:.6f} seconds")
        return result

    return wrapper


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

    _max = max(_max, 4000)

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

    def init_population_with_sgs(pop_size):
        population = []
        for _ in range(pop_size):
            individual = generate_feasible_individual(processes, stock, _max)
            population.append(individual)
        return population

    def is_valid_gene(gene, incomplete_dna):
        stock_copy = stock.copy()
        current_stock = get_stock_after_individual(
            incomplete_dna, processes, stock_copy
        )
        if (
            can_run_task(current_stock, processes[gene]["need"])
            and len(incomplete_dna) < _max
        ):
            return True
        return False

    def crossover(individual1, individual2):
        """Crossover two individuals."""
        child = []
        # group by genes individual1 and individual2
        count_individual1 = Counter(individual1)
        count_individual2 = Counter(individual2)
        for _ in range(int(max(len(individual1), len(individual2)))):
            for gene, count in count_individual1.items():
                if count > 0 and is_valid_gene(gene, child):
                    child.append(gene)
                    count_individual1[gene] -= 1
                    break
            for gene, count in count_individual2.items():
                if count > 0 and is_valid_gene(gene, child):
                    child.append(gene)
                    count_individual2[gene] -= 1
                    break
        # Check if the child is valid
        if len(child) >= _min:
            return child, True
        else:
            # If the child is not valid, return the first parent
            # and set is_child to False
            return individual1[:], False

    ga = GeneticAlgorithm(
        population_size=500,
        crossover_rate=0.7,
        elite_rate=0.05,
        selection_rate=0.5,
        mutation_rate=0.02,
        genes=list(processes.keys()),
        fitness_function=fitness_function,
        init_population=init_population_with_sgs,
        valid_gene=is_valid_gene,
        # crossover=crossover,
        generations=100,
        min_dna_length=_min,
        max_dna_length=_max,
    )

    best = ga.run()

    print("Best fitness:", fitness_function(best))
    # print("Best individual:", best, len(best))

    print(
        "Stock after best individual:",
        get_stock_after_individual(best, processes, stock.copy()),
    )
