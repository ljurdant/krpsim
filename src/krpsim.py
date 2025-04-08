#!/usr/bin/env python3

from _types import Process
from typing import List
from genetic_algorithm import GeneticAlgorithm
from parser import parse
import sys

from matplotlib import pyplot as plt


def do_process(
    process_name: str, processes: dict[str, Process], stock: dict[str, int]
) -> int:
    """Do a process and update the stock."""
    # Check if we have enough resources
    process = processes[process_name]

    for resource, amount in process["need"].items():
        if stock.get(resource, 0) < amount:
            return process["time"], False

    # Update the stock
    for resource, amount in process["need"].items():
        stock[resource] -= amount

    # Add the result to the stock
    for resource, amount in process["result"].items():
        stock[resource] = stock.get(resource, 0) + amount

    return process["time"], True


def get_score(
    individual: List[Process], processes, stock: dict[str, int], optimize: List[str]
) -> float:
    stop_index = individual.index("stop") if "stop" in individual else -1
    if stop_index != -1:
        individual = individual[:stop_index]

    total_time = 0
    failed_steps = 0

    for process_name in individual:
        time_taken, success = do_process(process_name, processes, stock)
        total_time += time_taken
        if not success:
            failed_steps += 1

    target_resources_count = sum(stock.get(resource, 0) for resource in optimize)

    if "time" in optimize:
        return target_resources_count / total_time if total_time > 0 else 0
    else:
        return target_resources_count


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

    if max > 20000:
        max = 20000

    def fitness_function(individual):
        stock_copy = stock.copy()
        processes_copy = processes.copy()
        optimize_copy = optimize.copy()
        return get_score(individual, processes_copy, stock_copy, optimize_copy)

    ga = GeneticAlgorithm(
        population_size=100,
        crossover_rate=0.9,
        elite_rate=0.05,
        selection_rate=0.6,
        mutation_rate=0.02,
        genes=list(processes.keys()),
        fitness_function=fitness_function,
        generations=100,
    )
    best = ga.run(max_dna_length=max, min_dna_length=min)

    print("Best fitness:", fitness_function(best))
    best = best[: best.index("stop")] if "stop" in best else best
    print("Best individual:", best)
