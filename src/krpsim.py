from _types import Process
from typing import List
from genetic_algorithm import GeneticAlgorithm


def parse() -> tuple[dict[str, int], dict[str, Process], List[str]]:
    # ikea demo
    stock = {"planche": 7}
    processes = {
        "do_montant": {"need": {"planche": 1}, "result": {"montant": 1}, "time": 15},
        "do_fond": {"need": {"planche": 2}, "result": {"fond": 1}, "time": 20},
        "do_etagere": {"need": {"planche": 1}, "result": {"etagere": 1}, "time": 10},
        "do_armoire_ikea": {
            "need": {"montant": 2, "fond": 1, "etagere": 3},
            "result": {"armoire": 1},
            "time": 30,
        },
    }
    optimize = ["time", "armoire"]

    return stock, processes, optimize


def do_process(
    process_name: str, processes: dict[str, Process], stock: dict[str, int]
) -> int:
    """Do a process and update the stock."""
    # Check if we have enough resources
    process = processes[process_name]
    print(f"Doing process {process_name} with stock {stock}")
    # print(f"Need {process['need']} for process {process_name}")
    # print(f"Result {process['result']} for process {process_name}")
    # print(f"Time {process['time']} for process {process_name}")
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
    stock, processes, optimize = parse()
    print("Stock:", stock)

    def fitness_function(individual):
        return get_score(individual, processes, stock, optimize)

    ga = GeneticAlgorithm(
        genes=list(processes.keys()), fitness_function=fitness_function
    )
    best = ga.run(dna_length=6)

    print("Best fitness:", fitness_function(best))

    print("Best individual:", best)
