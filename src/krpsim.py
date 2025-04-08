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


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python main.py <path_to_krpsim_config>")
        sys.exit(1)

    config_file = sys.argv[1]
    stock, processes, optimize = parse(config_file)

    def fitness_function(individual):
        stock_copy = stock.copy()
        processes_copy = processes.copy()
        optimize_copy = optimize.copy()
        return get_score(individual, processes_copy, stock_copy, optimize_copy)

    ga = GeneticAlgorithm(
        population_size=200,
        crossover_rate=0.9,
        elite_rate=0.05,
        selection_rate=0.6,
        mutation_rate=0.02,
        genes=list(processes.keys()),
        fitness_function=fitness_function,
        generations=100,
    )
    best = ga.run(dna_length=100)

    print("Best fitness:", fitness_function(best))

    print("Best individual:", best)
