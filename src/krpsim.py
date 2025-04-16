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


def final_format(individual: List[dict[str, int]], processes, stock) -> List[str]:
    """
    Convert the individual to a final format.
    """
    final_individual = []
    current_cycle = 0
    i = 0
    while i < len(individual):
        for _ in range(individual[i]["amount"]):
            final_individual.append(f"{current_cycle}:{individual[i]['process']}")
        resource_diffs = []
        resoource_diff = do_process(individual[i]["process"], processes, stock)

        for consumed_resource, amount_consumed in resoource_diff["consumed"].items():
            stock[consumed_resource] = (
                stock.get(consumed_resource, 0)
                - amount_consumed * individual[i]["amount"]
            )
        for produced_resource, amount_produced in resoource_diff["produced"].items():
            resoource_diff["produced"][produced_resource] *= individual[i]["amount"]
        resource_diffs.append(resoource_diff)

        can_run = True
        max_time = processes[individual[i]["process"]]["time"]
        while i + 1 < len(individual) and can_run:
            resoource_diff = do_process(individual[i + 1]["process"], processes, stock)
            can_run = True

            if len(resoource_diff["consumed"]) == 0:
                i += 1
                can_run = False
                break
            for resource, amount in resoource_diff["consumed"].items():
                if (stock.get(resource, 0) - amount * individual[i + 1]["amount"]) < 0:
                    can_run = False
            if can_run:
                for consumed_resource, amount_consumed in resoource_diff[
                    "consumed"
                ].items():
                    stock[consumed_resource] = (
                        stock.get(consumed_resource, 0)
                        - amount_consumed * individual[i + 1]["amount"]
                    )
                for produced_resource, amount_produced in resoource_diff[
                    "produced"
                ].items():
                    resoource_diff["produced"][produced_resource] *= individual[i + 1][
                        "amount"
                    ]
                resource_diffs.append(resoource_diff)
                if processes[individual[i + 1]["process"]]["time"] > max_time:
                    max_time = processes[individual[i + 1]["process"]]["time"]
                for _ in range(individual[i + 1]["amount"]):
                    final_individual.append(
                        f"{current_cycle}:{individual[i + 1]['process']}"
                    )
                i += 1
            else:
                i += 1
                break
        current_cycle += max_time
        for resource_diff in resource_diffs:
            for produced_resource, amount_produced in resource_diff["produced"].items():
                stock[produced_resource] = (
                    stock.get(produced_resource, 0) + amount_produced
                )
        if i + 1 == len(individual) and can_run == True:
            i += 1
    return final_individual


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
    min_length=1,
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
    for opt in [opt for opt in optimize if opt != "time"]:
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

    for opt in [opt for opt in optimize if opt != "time"]:
        if "time" in optimize:
            for process in processes.values():
                if opt in process["result"].keys():
                    hierarchy_dict[opt] = 2 * process["time"]
        else:
            hierarchy_dict[opt] = 2
    return hierarchy_dict


def get_score(
    individual: List[dict[str, int]],
    processes,
    stock: dict[str, int],
    optimize: List[str],
    hierarchy: dict[str, List[str]],
) -> float:

    init_resources_count = sum(
        stock.get(resource, 0) * hierarchy.get(resource, 0) for resource in stock.keys()
    )
    # init_resources_count = sum(stock.get(resource, 0) for resource in optimize)

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
        if real_amount > 0:
            total_time += processes[process_name]["time"]

    # resources_count = sum(stock.get(resource, 0) for resource in optimize)
    resources_count = sum(
        stock.get(resource, 0) * hierarchy.get(resource, 0) for resource in stock.keys()
    )

    resource_diff = resources_count - init_resources_count

    if "time" in optimize:
        return resource_diff / total_time if total_time > 0 else 0
    else:
        return resource_diff


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

        for consumed_resource, amount_consumed in resource_diff["consumed"].items():
            stock[consumed_resource] = (
                stock.get(consumed_resource, 0) - amount_consumed * real_amount
            )
        for produced_resource, amount_produced in resource_diff["produced"].items():
            stock[produced_resource] = (
                stock.get(produced_resource, 0) + amount_produced * real_amount
            )
        total_time += processes[process_name]["time"]
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
                    max_length += amount
                else:
                    min_length, max_length = get_min_max_gene_length(
                        min_length + 1, max_length * amount, processes, resource, stock
                    )
            break

    return min_length, max_length


if __name__ == "__main__":

    if len(sys.argv) < 3:
        print("Usage: python main.py <path_to_krpsim_config> <max_execution_time(sec)>")
        sys.exit(1)

    config_file = sys.argv[1]
    max_execution_time = int(sys.argv[2])
    if max_execution_time < 1:
        print("max_execution_time must be greater than 0")
        sys.exit(1)
    stock, processes, optimize = parse(config_file)

    _min = 1
    _max = 2
    for opt in optimize:
        tmp_min, tmp_max = get_min_max_gene_length(0, 1, processes, opt, stock)
        _max = _max + tmp_max
    _min = 1
    _max = min(_max, 50000)

    hierarchy = get_resource_hierarchy(processes, optimize)

    def fitness_function(individual):
        stock_copy = stock.copy()

        return get_score(individual, processes, stock_copy, optimize, hierarchy)

    def init_population_with_sgs(pop_size):
        population = []
        rand_portion = 1
        rand_pop_size = int(pop_size * rand_portion)

        for _ in range(rand_pop_size):
            individual = generate_random_individual(
                processes, _max, _min, max(_max // 100, 1)
            )
            population.append(individual)

        return population

    ga = GeneticAlgorithm(
        population_size=200,
        crossover_rate=0.7,
        elite_rate=0.01,
        selection_rate=0.7,
        mutation_rate=0.005,
        genes=list(processes.keys()),
        fitness_function=fitness_function,
        init_population=init_population_with_sgs,
        time_limit=max_execution_time,
        parent_selection_type="trounament",
        selection_pressure=2,
        tournament_probability=0.2,
        crossover_point="uniform",
        hyper_mutation_rate=0.0005,
        hyper_change_frequency=3,
        hyper_tournament_probability=0.9,
        hyper_selection_pressure=3,
        hyper_numb_generation=3,
    )

    best, fitnesses = ga.run()

    fitness = fitness_function(best)
    print("Best fitness:", fitness)

    valid_best = trim_invalid(best, processes, stock.copy())
    print(
        "Stock after best individual:",
        get_stock_after_individual(valid_best, processes, stock.copy()),
    )
    print("Best individual:", valid_best)

    now = datetime.now()

    current_time = now.strftime("%Y%m%d_%H:%M:%S")
    print("Time:", current_time)

    to_save = final_format(valid_best, processes, stock)
    print("Save results? (y/n)")
    save = input()

    if save == "y":
        with open(
            f"../results/{config_file.split('/')[-1]}_{fitness*100:.2f}_{current_time}.json",
            "w",
        ) as f:
            for to_save_line in to_save:
                f.write(f"{to_save_line}\n")
    plt.plot(fitnesses)
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("Fitness over generations")
    plt.show()
