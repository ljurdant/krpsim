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
            return 0, False

    # Update the stock
    for resource, amount in process["need"].items():
        stock[resource] -= amount

    # Add the result to the stock
    for resource, amount in process["result"].items():
        stock[resource] = stock.get(resource, 0) + amount

    return process["time"], True


# def compute_hierarchy(processes: dict[str, dict], root_weights=None):
#     if root_weights is None:
#         root_weights = {}

#     resource_weights = {}

#     def get_weight(resource):
#         if resource in resource_weights:
#             return resource_weights[resource]
#         weight = 1.0  # base weight

#         # Check which processes produce this resource
#         producing_procs = [p for p in processes.values() if resource in p["result"]]

#         for proc in producing_procs:
#             time = proc["time"]
#             for need_res, qty in proc["need"].items():
#                 weight += get_weight(need_res) * qty
#             weight += time  # more time = more value

#         resource_weights[resource] = weight
#         return weight

#     # Start by computing weight for all result resources
#     for proc in processes.values():
#         for res in proc["result"]:
#             get_weight(res)

#     # Normalize
#     max_weight = max(resource_weights.values())
#     for res in resource_weights:
#         resource_weights[res] /= max_weight

#     return resource_weights


def get_resource_hierarchy(
    processes: dict[str, Process], optimize: List[str]
) -> dict[str, List[str]]:

    processes = processes.copy()
    # Remove the stop process
    optimize_resources = [resource for resource in optimize if resource != "time"]

    resources = set(
        [key for process in processes.values() for key in process["need"].keys()]
        + [key for process in processes.values() for key in process["result"].keys()]
    )
    hierarchy_dict = {}
    for resource in optimize_resources:
        resources.remove(resource)
        hierarchy_dict[resource] = 1

    while len(resources):
        tmp_resources = resources.copy()
        for resource in tmp_resources:
            needed = 0
            b = False
            for process in processes.values():
                if (
                    sum(
                        key in hierarchy_dict.keys() for key in process["result"].keys()
                    )
                    > 0
                ):
                    if resource in process["need"]:
                        needed = 0.5 * max(
                            hierarchy_dict[key] if key in hierarchy_dict.keys() else 0
                            for key in process["result"].keys()
                        )
                        b = True
                        if resource in hierarchy_dict.keys():

                            hierarchy_dict[resource] = max(
                                hierarchy_dict[resource], needed
                            )
                        else:
                            hierarchy_dict[resource] = needed
            if b:
                resources.remove(resource)

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
        if not success:
            break

    resources_count = sum(
        [stock.get(resource) * hierarchy.get(resource, 0) for resource in stock.keys()]
    )
    target_resources_count = sum(stock.get(resource, 0) for resource in optimize)
    # resources_count += target_resources_count

    if "time" in optimize:
        return target_resources_count / total_time if total_time > 0 else 0
    else:
        return (
            target_resources_count * 5
            + valid_count * 0.5
            + len([resource for resource, amount in stock.items() if amount > 0]) * 0.2
        )


def get_stock_after_individual(
    individual: List[Process], processes, stock: dict[str, int], optimize: List[str]
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

    hierarchy = get_resource_hierarchy(processes, optimize)

    print("Resource hierarchy:", hierarchy)

    min = 0
    max = 0
    for opt in optimize:
        tmp_min, tmp_max = get_min_max_gene_length(0, 1, processes, opt, stock)
        min = min + tmp_min
        max = max + tmp_max

    # min = 1000
    if max > 2000:
        max = 2000

    def fitness_function(individual):
        stock_copy = stock.copy()
        processes_copy = processes.copy()
        optimize_copy = optimize.copy()
        return get_score(
            individual, processes_copy, stock_copy, optimize_copy, hierarchy
        )

    def get_valid_sequence(processes_names: List[str]) -> List[str]:
        """Get a valid sequence of processes to do."""
        stock_copy = stock.copy()
        # Get the valid processes
        valid_processes = []
        for process_name in processes_names:
            # Check if we have enough resources
            if all(
                stock_copy.get(resource, 0) >= amount
                for resource, amount in processes[process_name]["need"].items()
            ):
                valid_processes.append(process_name)
                # remove the resources from the stock_copy
                for resource, amount in processes[process_name]["need"].items():
                    stock_copy[resource] -= amount
                # add the results to the stock_copy
                for resource, amount in processes[process_name]["result"].items():
                    stock_copy[resource] = stock_copy.get(resource, 0) + amount
        return valid_processes

    ga = GeneticAlgorithm(
        population_size=5000,
        crossover_rate=0.7,
        elite_rate=0.05,
        selection_rate=0.5,
        mutation_rate=0.1,
        genes=list(processes.keys()),
        fitness_function=fitness_function,
        get_valid_sequence=get_valid_sequence,
        generations=10,
    )

    best = ga.run(max_dna_length=max, min_dna_length=min)

    print("Best fitness:", fitness_function(best))
    print("Best individual:", best, len(best))
    print(
        "Stock after best individual:",
        get_stock_after_individual(best, processes, stock.copy(), optimize),
    )
