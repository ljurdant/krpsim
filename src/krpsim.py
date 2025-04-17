#!/usr/bin/env python3

from _types import Process
from typing import List
from genetic_algorithm import GeneticAlgorithm
import random
from parser import parse
import sys
import json
import os

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

        chromosome.append(
            {
                "amount": task_length,
                "process": chosen_task,
                "parallel": bool(random.randint(0, 1)),
            }
        )
        total_length += task_length

    return chromosome


def get_resource_hierarchy(
    processes: dict[str, dict], optimize: dict[str, int], level=1, exclude={}
) -> dict[str, float]:
    hierarchy = {}

    for process in processes.values():
        result_keys = set(process["result"].keys())

        # If the process produces something already in the hierarchy, update its inputs
        for result in result_keys:
            if result in optimize:
                for need in process["need"]:
                    value = (
                        process["result"][result]
                        / process["need"][need]
                        * optimize[result]
                    )
                    hierarchy[need] = max(hierarchy.get(need, 0), value)
    for optimize_key in optimize.keys():
        exclude[optimize_key] = 1
    next_level = {key: hierarchy[key] for key in hierarchy.keys() if key not in exclude}
    if len(next_level):
        sub_hierarchy = get_resource_hierarchy(
            processes, next_level, level + 1, exclude=exclude
        )
        hierarchy.update(sub_hierarchy)
    return hierarchy


def run_individual(
    individual: List[dict[str, int]],
    processes,
    stock: dict[str, int],
):

    total_time = 0
    i = 0
    while i < len(individual):
        parallel_tasks = [individual[i]]
        i += 1
        while i < len(individual) and individual[i]["parallel"]:
            parallel_tasks.append(individual[i])
            i += 1
        time = 0
        real_amounts = []
        resource_diffs = []
        for gene in parallel_tasks:
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

            resource_diffs.append(resource_diff)
            real_amounts.append(real_amount)

            if real_amount > 0:
                time = max(time, processes[process_name]["time"])

            for consumed_resource, amount_consumed in resource_diff["consumed"].items():
                stock[consumed_resource] = (
                    stock.get(consumed_resource, 0) - amount_consumed * real_amount
                )

        for n in range(len(parallel_tasks)):
            for produced_resource, amount_produced in resource_diffs[n][
                "produced"
            ].items():
                stock[produced_resource] = (
                    stock.get(produced_resource, 0) + amount_produced * real_amounts[n]
                )
        total_time += time
    return total_time, stock


def get_score(
    individual: tuple[List[dict[str, int]], tuple[float, dict[str, int]]],
    stock: dict[str, int],
    optimize: List[str],
    hierarchy: dict[str, List[str]],
) -> float:

    init_resources_count = sum(stock.get(resource, 0) for resource in optimize)

    total_time, new_stock = individual[1]

    # resources_count = sum(stock.get(resource, 0) for resource in optimize)
    resources_count = sum(new_stock.get(resource, 0) for resource in optimize)

    resource_diff = resources_count - init_resources_count

    if "time" in optimize:
        return resource_diff / total_time if total_time > 0 else 0
    else:
        return resource_diff


def compare_scores(
    individual1: tuple[List[dict[str, int]], tuple[float, dict[str, int]]],
    individual2: tuple[List[dict[str, int]], tuple[float, dict[str, int]]],
    stock: dict[str, int],
    optimize: List[str],
    hierarchy: dict[str, List[str]],
):

    init_resources_count = sum(
        stock.get(resource, 0) * hierarchy.get(resource, 0) for resource in stock.keys()
    )
    init_optimize_count = sum(stock.get(resource, 0) for resource in optimize)

    total_time1, stock1 = individual1[1]
    total_time2, stock2 = individual2[1]

    optimize_diff1 = (
        sum(stock1.get(resource, 0) for resource in optimize) - init_optimize_count
    )

    optimize_diff2 = (
        sum(stock2.get(resource, 0) for resource in optimize) - init_optimize_count
    )

    if "time" in optimize:
        optimize_diff1 = optimize_diff1 / total_time1 if total_time1 > 0 else 0
        optimize_diff2 = optimize_diff2 / total_time2 if total_time2 > 0 else 0

    if optimize_diff1 != optimize_diff2:
        return optimize_diff1 - optimize_diff2
    else:
        resource_diff1 = (
            sum(
                stock1.get(resource, 0) * hierarchy.get(resource, 0)
                for resource in stock1.keys()
            )
            - init_resources_count
        )
        resource_diff2 = (
            sum(
                stock2.get(resource, 0) * hierarchy.get(resource, 0)
                for resource in stock2.keys()
            )
            - init_resources_count
        )

        if "time" in optimize:
            resource_diff1 = resource_diff1 / total_time1 if total_time1 > 0 else 0
            resource_diff2 = resource_diff2 / total_time2 if total_time2 > 0 else 0

        return resource_diff1 - resource_diff2


def trim_invalid(
    individual: List[dict[str, int]], processes, stock: dict[str, int]
) -> List[dict[str, int]]:
    valid_individual = []
    i = 0
    while i < len(individual):
        parallel_tasks = [individual[i]]
        i += 1
        while i < len(individual) and individual[i]["parallel"]:
            parallel_tasks.append(individual[i])
            i += 1

        real_amounts = []
        resource_diffs = []
        for gene in parallel_tasks:
            process_name = gene["process"]
            process_amount = gene["amount"]

            resource_diff = do_process(process_name, processes, stock)

            max_amount_possible = 0
            for resource, amount in resource_diff["consumed"].items():
                resource_possible = stock.get(resource, 0) // amount
                if max_amount_possible == 0:
                    max_amount_possible = resource_possible
                else:
                    max_amount_possible = min(max_amount_possible, resource_possible)

            real_amount = min(process_amount, max_amount_possible)
            resource_diffs.append(resource_diff)
            real_amounts.append(real_amount)

            for consumed_resource, amount_consumed in resource_diff["consumed"].items():
                stock[consumed_resource] = (
                    stock.get(consumed_resource, 0) - amount_consumed * real_amount
                )
        parallel = False
        for n in range(len(parallel_tasks)):
            for produced_resource, amount_produced in resource_diffs[n][
                "produced"
            ].items():
                stock[produced_resource] = (
                    stock.get(produced_resource, 0) + amount_produced * real_amounts[n]
                )
            if real_amounts[n] > 0:

                valid_individual.append(
                    {
                        "amount": real_amounts[n],
                        "process": parallel_tasks[n]["process"],
                        "parallel": parallel,
                    }
                )
                parallel = True
    valid_individual[0]["parallel"] = True
    return valid_individual


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

    min_gene_length = 1
    max_chromosome_length = 1
    for opt in optimize:
        tmp_min, tmp_max = get_min_max_gene_length(0, 1, processes, opt, stock)
        max_chromosome_length = max_chromosome_length + tmp_max
    max_chromosome_length = min(max_chromosome_length, 50000)
    max_gene_length = max(max_chromosome_length // 100, 1)

    hierarchy = get_resource_hierarchy(
        processes, {opt: 1 for opt in optimize if opt != "time"}
    )
    for opt in optimize:
        if opt in hierarchy:
            del hierarchy[opt]

    print("Hierarchy:", hierarchy)

    def fitness_function(individual):
        return get_score(individual, stock.copy(), optimize, hierarchy)

    def cmp_function(
        individual1: List[dict[str, int]], individual2: List[dict[str, int]]
    ):
        return compare_scores(
            individual1, individual2, stock.copy(), optimize, hierarchy
        )

    def get_extra_params(individual):
        """
        Get extra parameters for the individual.
        """
        individual_time, individual_stock = run_individual(
            individual, processes, stock.copy()
        )
        return (individual_time, individual_stock)

    def init_population_with_sgs(pop_size):
        population = []
        rand_portion = 1
        rand_pop_size = int(pop_size * rand_portion)

        for _ in range(rand_pop_size):
            individual = generate_random_individual(
                processes, max_chromosome_length, min_gene_length, max_gene_length
            )
            population.append((individual, get_extra_params(individual)))

        return population

    ga = GeneticAlgorithm(
        population_size=300,
        crossover_rate=0.7,
        elite_rate=0.01,
        selection_rate=0.7,
        mutation_rate=0.0005,
        genes=list(processes.keys()),
        fitness_function=fitness_function,
        init_population=init_population_with_sgs,
        time_limit=max_execution_time,
        parent_selection_type="random",
        selection_pressure=8,
        tournament_probability=0.7,
        crossover_point="single",
        hyper_mutation_rate=0.005,
        hyper_change_frequency=3,
        hyper_tournament_probability=0.5,
        hyper_selection_pressure=2,
        hyper_numb_generation=3,
        max_gene_length=max_gene_length,
        min_gene_length=min_gene_length,
        cmp_function=cmp_function,
        get_extra_params=get_extra_params,
        # generations=1,
    )

    best, fitnesses = ga.run()

    fitness = ga.fitness_function(best)
    print("Best fitness:", fitness)

    valid_best = trim_invalid(best[0], processes, stock.copy())

    now = datetime.now()

    current_time = now.strftime("%Y%m%d_%H:%M:%S")

    to_save = final_format(valid_best, processes, stock)
    # print(stock, optimize)
    # print(
    #     compare_scores(
    #         ("1", (30, {"energy": 45})),
    #         ("2", (30, {"cell": 2})),
    #         stock,
    #         optimize,
    #         hierarchy,
    #     )
    # )
    print("Save results? (y/n)")
    save = input()
    print(sum([process["amount"] for process in valid_best]))
    if save == "y":
        filename = (
            f"../results/{config_file.split('/')[-1]}_{fitness:.2f}_{current_time}.json"
        )
        with open(
            f"../results/{config_file.split('/')[-1]}_{fitness:.2f}_{current_time}.json",
            "w",
        ) as f:
            for to_save_line in to_save:
                f.write(f"{to_save_line}\n")

        print("Run verification? (y/n)")
        verify = input()
        if verify == "y":
            os.execve(
                f"./krpsim_verif.py",
                ["./krpsim_verif.py", config_file, filename],
                {},
            )
    plt.plot(fitnesses)
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("Fitness over generations")
    plt.show()
