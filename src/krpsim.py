#!/usr/bin/env python3

from _types import Process
from typing import List
from genetic_algorithm import GeneticAlgorithm
import random
from parser import parse
import sys

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


def get_score(
    individual: List[Process],
    processes,
    stock: dict[str, int],
    optimize: List[str],
) -> float:

    total_time = 0
    valid_count = 0
    for process_name in individual:
        time_taken, success = do_process(process_name, processes, stock)
        total_time += time_taken
        valid_count += 1
        if not success:
            break

    target_resources_count = sum(stock.get(resource, 0) for resource in optimize)

    if valid_count == 0:
        return -5
    if "time" in optimize:
        return target_resources_count / total_time if total_time > 0 else 0
    else:
        return target_resources_count * 10000 + len(
            [resource for resource, amount in stock.items() if amount > 0]
        )
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

    min = 0
    max = 0
    for opt in optimize:
        tmp_min, tmp_max = get_min_max_gene_length(0, 1, processes, opt, stock)
        min = min + tmp_min
        max = max + tmp_max

    # min = 1000
    if max > 2000:
        max = 2000

    print("Min gene length:", min)
    print("Max gene length:", max)

    def fitness_function(individual):
        stock_copy = stock.copy()
        processes_copy = processes.copy()
        optimize_copy = optimize.copy()
        return get_score(individual, processes_copy, stock_copy, optimize_copy)

    def init_population_with_sgs(pop_size):
        population = []
        for _ in range(pop_size):
            individual = generate_feasible_individual(processes, stock, max)
            population.append(individual)
        return population

    ga = GeneticAlgorithm(
        population_size=100,
        crossover_rate=0.7,
        elite_rate=0.05,
        selection_rate=0.5,
        mutation_rate=0.5,
        genes=list(processes.keys()),
        fitness_function=fitness_function,
        init_population=init_population_with_sgs,
        generations=100,
        min_dna_length=min,
        max_dna_length=max,
    )

    best = ga.run()

    print("Best fitness:", fitness_function(best))
    print("Best individual:", best, len(best))
    print(
        "Stock after best individual:",
        get_stock_after_individual(best, processes, stock.copy()),
    )
