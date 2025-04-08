from _types import Process
from typing import List
from genetic_algorithm import GeneticAlgorithm
from parser import parse
import sys


def do_process(
    process_name: str, processes: dict[str, Process], stock: dict[str, int]
) -> int:
    """Do a process and update the stock."""
    # Check if we have enough resources
    process = processes[process_name]

    for resource, amount in process["need"].items():
        if stock.get(resource, 0) < amount:
            return process["time"]

    # Update the stock
    for resource, amount in process["need"].items():
        stock[resource] -= amount

    # Add the result to the stock
    for resource, amount in process["result"].items():
        stock[resource] = stock.get(resource, 0) + amount

    return process["time"]


def get_score(
    individual: List[Process], processes, stock: dict[str, int], optimize: List[str]
) -> float:
    total_time = sum(
        do_process(process_name, processes, stock) for process_name in individual
    )
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
    print(stock, processes, optimize)

    def fitness_function(individual):
        stock_copy = stock.copy()
        processes_copy = processes.copy()
        optimize_copy = optimize.copy()
        return get_score(individual, processes_copy, stock_copy, optimize_copy)

    ga = GeneticAlgorithm(
        genes=list(processes.keys()), fitness_function=fitness_function
    )
    best = ga.run(dna_length=8)

    print("Best fitness:", fitness_function(best))

    print("Best individual:", best)
