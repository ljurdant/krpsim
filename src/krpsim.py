#!/usr/bin/env python3

from _types import Process
from typing import List
from genetic_algorithm import GeneticAlgorithm
import random
from parser import parse
import sys
import json

import time

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
) -> dict[str, dict[str, int]]:
    """Do a process and update the stock."""
    # Check if we have enough resources
    process = processes[process_name]
    result = {"consumed": {}, "produced": {}}
    for resource, amount in process["need"].items():
        if stock.get(resource, 0) < amount:
            return result

    for consumed_resource, amount_consumed in process["need"].items():
        result["consumed"][consumed_resource] = amount_consumed

    for produced_resource, amount_produced in process["result"].items():
        result["produced"][produced_resource] = amount_produced

    return result


def generate_feasible_individual(
    processes: dict[str, Process],
    initial_stock: dict[str, int | str],
    max_length=30,
) -> List[dict[str, int]]:
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
    length,
    min_length: int = 0,
    max_length=30,
) -> List[dict[str, int]]:

    # task_names = list(processes.keys())  # Genes
    # length = random.randint(min_length, max_length)
    # chromosome = [random.choice(task_names) for _ in range(length)]
    chromosome = []
    total_length = 0
    while total_length < length:
        # Randomly select one feasible task
        chosen_task = random.choice(list(processes.keys()))

        # Append to chromosome
        task_length = random.randint(min_length, max_length)
        task_length = min(task_length, length - total_length)

        chromosome.append({"amount": task_length, "process": chosen_task})
        total_length += task_length
    if sum(gene["amount"] for gene in chromosome) != length:
        print("chromosome", chromosome)
        raise Exception("stop")
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
    individual: List[dict[str, int]],
    processes,
    stock: dict[str, int],
    optimize: List[str],
) -> float:

    total_time = 0
    for gene in individual:
        process_name = gene["process"]
        process_amount = gene["amount"]

        resource_diff = do_process(process_name, processes, stock)

        min_amount_possible = 0
        for resource, amount in resource_diff["consumed"].items():
            resource_possible = stock.get(resource, 0) // amount
            if min_amount_possible == 0:
                min_amount_possible = resource_possible
            else:
                min_amount_possible = min(min_amount_possible, resource_possible)
        real_amount = min(process_amount, min_amount_possible)

        for consumed_resource, amount_consumed in resource_diff["consumed"].items():
            stock[consumed_resource] = (
                stock.get(consumed_resource, 0) - amount_consumed * real_amount
            )
        for produced_resource, amount_produced in resource_diff["produced"].items():
            stock[produced_resource] = (
                stock.get(produced_resource, 0) + amount_produced * real_amount
            )
        total_time += processes[process_name]["time"]

        # print(process_name, stock)

    # resources_count = sum(
    #     ((stock.get(resource, 0) * hierarchy[resource])) for resource in stock.keys()
    # )
    resources_count = sum(
        stock.get(resource, 0) for resource in stock.keys() if resource in optimize
    )

    # resource_diff = resources_count - initial_resources_count
    # print(resources_count)

    # if "time" in optimize:
    return resources_count / total_time if total_time > 0 else 0
    # else:
    # return resources_count  # + valid_count / len(individual)
    # I need to add valid_count to the score or something to reward good ressources collected


def trim_invalid(
    individual: List[dict[str, int]], processes, stock: dict[str, int]
) -> List[dict[str, int]]:
    valid_individual = []
    total_time = 0

    for gene in individual:
        process_name = gene["process"]
        process_amount = gene["amount"]

        resource_diff = do_process(process_name, processes, stock)

        min_amount_possible = 0
        for resource, amount in resource_diff["consumed"].items():
            resource_possible = stock.get(resource, 0) // amount
            if min_amount_possible == 0:
                min_amount_possible = resource_possible
            else:
                min_amount_possible = min(min_amount_possible, resource_possible)
        real_amount = min(process_amount, min_amount_possible)

        for (consumed_resource, amount_consumed), (
            produced_resource,
            amount_produced,
        ) in zip(resource_diff["consumed"].items(), resource_diff["produced"].items()):
            stock[consumed_resource] = (
                stock.get(consumed_resource, 0) - amount_consumed * real_amount
            )
            stock[produced_resource] = (
                stock.get(produced_resource, 0) + amount_produced * real_amount
            )

        total_time += real_amount * processes[process_name]["time"]
        if real_amount > 0:
            valid_individual.append(
                {
                    "process": process_name,
                    "amount": real_amount,
                }
            )

    return valid_individual


def get_stock_after_individual(
    individual: List[dict[str, int]], processes, stock: dict[str, int]
) -> dict[str, int]:

    for process in individual:
        for i in range(process["amount"]):
            resource_diff = do_process(process["process"], processes, stock)
            for consumed_resource, amount_consumed in resource_diff["consumed"].items():
                stock[consumed_resource] = (
                    stock.get(consumed_resource, 0) - amount_consumed
                )
            for produced_resource, amount_produced in resource_diff["produced"].items():
                stock[produced_resource] = (
                    stock.get(produced_resource, 0) + amount_produced
                )

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

    _min = 1
    _max = 2
    for opt in optimize:
        tmp_min, tmp_max = get_min_max_gene_length(0, 1, processes, opt, stock)
        _max = _max + tmp_max
    _min = 1
    _max = min(_max, 50000)

    def fitness_function(individual):
        stock_copy = stock.copy()
        processes_copy = processes.copy()
        optimize_copy = optimize.copy()
        return get_score(individual, processes_copy, stock_copy, optimize_copy)

    def init_population_with_sgs(pop_size):
        population = []
        rand_portion = 1
        rand_pop_size = int(pop_size * rand_portion)
        # feasible_pop_size = pop_size - rand_pop_size

        # for _ in range(feasible_pop_size):
        #     individual = generate_feasible_individual(processes, stock, _max)
        #     population.append(individual)

        for _ in range(rand_pop_size):
            individual = generate_random_individual(processes, _max, _min, _max // 10)
            population.append(individual)

        return population

    ga = GeneticAlgorithm(
        population_size=100,
        crossover_rate=0.7,
        elite_rate=0.01,
        selection_rate=0.7,
        mutation_rate=0.01,
        genes=list(processes.keys()),
        fitness_function=fitness_function,
        init_population=init_population_with_sgs,
        generations=1000,
        time_limit=30,
        parent_selection_type="random",
        crossover_point="single",
        min_series_length=_min,
        selection_pressure=8,
        tournament_probability=0.9,
        # max_series_length=_max // 10,
    )

    best, fitnesses = ga.run()

    fitness = fitness_function(best)
    print("Best fitness:", fitness)

    # print("Best individual:", best, len(best))
    valid_best = trim_invalid(best, processes, stock.copy())
    print(
        "Stock after best individual:",
        get_stock_after_individual(valid_best, processes, stock.copy()),
    )

    now = datetime.now()

    current_time = now.strftime("%Y%m%d_%H:%M:%S")
    print("Time:", current_time)

    print("Save results? (y/n)")
    save = input()
    if save == "y":
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
