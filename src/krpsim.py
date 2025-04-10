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
    stop_index = individual.index("stop") if "stop" in individual else -1
    if stop_index != -1:
        individual = individual[:stop_index]

    total_time = 0
    failed_steps = 0
    for process_name in individual:
        time_taken, success = do_process(process_name, processes, stock)
        total_time += time_taken
        failed_steps += int(not success)

    resources_count = sum(
        ((stock.get(resource, 0) * hierarchy[resource])) for resource in stock.keys()
    )

    if "time" in optimize:
        return resources_count / total_time if total_time > 0 else 0
    else:
        return resources_count


def get_stock_after_individual(
    individual: List[Process], processes, stock: dict[str, int], optimize: List[str]
) -> dict[str, int]:
    stop_index = individual.index("stop") if "stop" in individual else -1
    if stop_index != -1:
        individual = individual[:stop_index]

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

    hierarchy = get_resource_hierarchy(processes, optimize)
    print("Resource hierarchy:", hierarchy)
    min = 0
    max = 0
    for opt in optimize:
        tmp_min, tmp_max = get_min_max_gene_length(0, 1, processes, opt, stock)
        min = min + tmp_min
        max = max + tmp_max

    if max > 2000:
        max = 2000

    def fitness_function(individual):
        stock_copy = stock.copy()
        processes_copy = processes.copy()
        optimize_copy = optimize.copy()
        return get_score(
            individual, processes_copy, stock_copy, optimize_copy, hierarchy
        )

    ga = GeneticAlgorithm(
        population_size=500,
        crossover_rate=0.9,
        elite_rate=0.05,
        selection_rate=0.6,
        mutation_rate=0.02,
        genes=list(processes.keys()),
        fitness_function=fitness_function,
        generations=300,
    )

    best = ga.run(max_dna_length=max, min_dna_length=min)

    print("Best fitness:", fitness_function(best))
    best = best[: best.index("stop")] if "stop" in best else best
    print("Best individual:", best, len(best))
    print(
        "Stock after best individual:",
        get_stock_after_individual(best, processes, stock.copy(), optimize),
    )
