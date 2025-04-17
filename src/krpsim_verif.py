#!/usr/bin/env python3
import sys
import re
from collections import defaultdict
import argparse

from parser import parse


def can_run_tasks_in_parallel(stock, processes, task_names):
    """
    Checks if all tasks in task_names can start in parallel given 'stock'.
    For a *strict* version of parallel resource checking, we sum up
    the needed resources for all tasks, then see if stock can cover it.

    Alternatively, if you want to allow partial concurrency or some tasks
    to run first, you would need a more complex scheduling check.
    """

    # Also track each task's time for computing max duration
    max_duration = 0
    # Sum up total needs across all tasks
    total_needs = defaultdict(int)
    for name in task_names:
        proc = processes[name]
        for rsrc, needed_qty in proc["need"].items():
            total_needs[rsrc] += needed_qty

        # Track max
        if proc["time"] > max_duration:
            max_duration = proc["time"]

    # Now check if stock can handle that
    for rsrc, needed_qty in total_needs.items():
        if stock.get(rsrc, 0) < needed_qty:
            return max_duration, False
    return max_duration, True


def run_tasks_in_parallel(stock, processes, task_names):
    """
    Deduct the resources for all tasks, then add their results.
    Because they're "parallel," we do it in two steps:
      1) Summation of all needs -> deduct from stock
      2) Summation of all results -> add to stock
    """
    # Sum up total needs and results
    total_needs = defaultdict(int)
    total_results = defaultdict(int)

    # Also track each task's time for computing max duration
    max_duration = 0

    for name in task_names:
        proc = processes[name]
        # Track max
        if proc["time"] > max_duration:
            max_duration = proc["time"]

        # Accumulate needs
        for rsrc, needed_qty in proc["need"].items():
            total_needs[rsrc] += needed_qty

        # Accumulate results
        for rsrc, res_qty in proc["result"].items():
            total_results[rsrc] += res_qty

    # Deduct needs
    for rsrc, needed_qty in total_needs.items():
        stock[rsrc] = stock.get(rsrc, 0) - needed_qty

    # Add results
    for rsrc, res_qty in total_results.items():
        stock[rsrc] = stock.get(rsrc, 0) + res_qty

    return max_duration


def main():
    """
    Usage: python krpsim_verif.py <config_file> <tasks_file>

    The tasks_file must have lines:
        <cycle>:<task_name>

    If multiple tasks share the same <cycle>, we treat them as parallel tasks.
    We sum their resource needs, confirm that the stock can handle all at once,
    then we 'run' them and advance the 'real cycle' by the maximum time among them.

    If we see a subsequent group with a <cycle> smaller than our 'real cycle',
    we print an error and stop.
    """
    parser = argparse.ArgumentParser(description="krpsim verification script")
    parser.add_argument("config_file", help="Path to the krpsim config file")
    parser.add_argument("tasks_file", help="Path to the JSON file describing tasks")
    parser.add_argument(
        "-c",
        "--cycles",
        type=int,
        default=None,
        help="Number of cycles to run until stopping. If omitted, we run through the file exactly once.",
    )
    args = parser.parse_args()

    # 1) Parse the KRPSim config
    try:
        stock, processes, optimize = parse(args.config_file)
    except ValueError as e:
        print(f"Error parsing file: {e}", file=sys.stderr)
        sys.exit(1)

    # 2) Read the tasks from the <tasks_file>, line by line
    line_regex = re.compile(r"^(\d+)\s*:\s*(.+)$")
    tasks = []
    with open(args.tasks_file, "r") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            match = line_regex.match(line)
            if not match:
                print(f"Error: Malformed line (expected '<cycle>:<task>'): {line}")
                sys.exit(1)

            cycle_str, task_name = match.groups()
            cycle = int(cycle_str)
            task_name = task_name.strip()

            # If the task is not known, it's possible user typed an invalid process name
            if task_name not in processes:
                print(f"Error: Task '{task_name}' not found in processes.")
                sys.exit(1)

            tasks.append((cycle, task_name))

    # 3) Sort tasks by cycle (if not guaranteed sorted)
    tasks.sort(key=lambda x: x[0])

    # 4) Group tasks by cycle so that all tasks with the same cycle run in parallel
    from itertools import groupby

    current_real_cycle = 0
    cycle_limit = args.cycles
    grouped = [(key, list(group)) for key, group in groupby(tasks, key=lambda x: x[0])]
    # 5) Process each group in ascending cycle order
    if not cycle_limit:
        for cycle, tasks_in_group in grouped:
            # If the user tries to schedule tasks at a cycle < current_real_cycle, that's invalid
            if cycle < current_real_cycle:
                print(
                    f"Error: Task(s) scheduled at cycle {cycle}, but the current real time is {current_real_cycle}."
                )
                print("Cannot go back in time. Verification failed.")
                sys.exit(1)

            # Gather just the task names in this group
            task_names = [task_name for _, task_name in tasks_in_group]

            # Check if we have enough resources to run them in parallel
            if not can_run_tasks_in_parallel(stock, processes, task_names):
                print(
                    f"Error: Not enough resources to run tasks {task_names} in parallel at cycle {cycle}."
                )
                continue

            # If we can run them, we do so, and move the real cycle forward
            # to cycle + the max duration among them.
            max_dur = run_tasks_in_parallel(stock, processes, task_names)
            current_real_cycle = cycle + max_dur
    else:
        beginning_cycle = 0
        new_duration = 0
        print(f"Running for {cycle_limit} cycles.")
        while current_real_cycle + new_duration < cycle_limit:
            print(f"Current real cycle: {current_real_cycle}")
            for cycle, tasks_in_group in grouped:
                # If the user tries to schedule tasks at a cycle < current_real_cycle, that's invalid
                if cycle + beginning_cycle < current_real_cycle:
                    print(
                        f"Error: Task(s) scheduled at cycle {cycle + beginning_cycle}, but the current real time is {current_real_cycle}."
                    )
                    print("Cannot go back in time. Verification failed.")
                    sys.exit(1)

                # Gather just the task names in this group
                task_names = [task_name for _, task_name in tasks_in_group]

                # Check if we have enough resources to run them in parallel
                new_duration, can_run = can_run_tasks_in_parallel(
                    stock, processes, task_names
                )
                # If we reached the limit, we stop
                if cycle + beginning_cycle + new_duration > cycle_limit:
                    break
                if not can_run:
                    print(
                        f"Error: Not enough resources to run tasks {task_names} in parallel at cycle {cycle + beginning_cycle}."
                    )
                    continue

                # If we can run them, we do so, and move the real cycle forward
                # to cycle + the max duration among them.
                max_dur = run_tasks_in_parallel(stock, processes, task_names)
                current_real_cycle = cycle + beginning_cycle + max_dur
            beginning_cycle = current_real_cycle

    # 6) End: print final stock and final real cycle
    print("\n===== VERIFICATION SUCCESS =====")
    print("Final stock:")
    for rsrc, qty in stock.items():
        print(f"  {rsrc}: {qty}")

    print(f"Final real cycle: {current_real_cycle}")
    print("================================")


if __name__ == "__main__":
    main()
