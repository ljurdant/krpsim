import sys
import re
import ast  # for safely parsing the Python list from the tasks file

from krpsim import can_run_task, run_task
from parser import parse


if __name__ == "__main__":
    """
    Usage:
        python verify_solution.py <config_file> <tasks_file>

    Where <tasks_file> is a file containing a Python list of tasks, e.g.:
        ["task1", "task2", "task3"]

    Example:
        python verify_solution.py ikea.krp my_tasks.txt
    """
    if len(sys.argv) < 3:
        print("Usage: python verify_solution.py <config_file> <tasks_file>")
        sys.exit(1)

    config_file = sys.argv[1]
    tasks_file = sys.argv[2]

    # 1) Parse KRPSim config
    stock, processes, optimize = parse(config_file)

    # 2) Load the task list from the second file
    with open(tasks_file, "r") as f:
        tasks_content = f.read().strip()
    try:
        # Convert the text (like '["task1", "task2"]') into a Python list
        task_sequence = ast.literal_eval(tasks_content)
        if not isinstance(task_sequence, list):
            print("[ERROR] The tasks file does not contain a valid Python list.")
            sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Unable to parse the tasks file: {e}")
        sys.exit(1)

    # We'll track total time
    total_time = 0

    # 3) Run tasks in order
    for task_name in task_sequence:
        if task_name not in processes:
            print(f"[WARNING] Task '{task_name}' not defined in processes. Skipping.")
            continue
        proc = processes[task_name]
        if can_run_task(stock, proc["need"]):
            run_task(stock, proc)
            total_time += proc["time"]
        else:
            print(f"[WARNING] Insufficient resources for '{task_name}'. Skipping.")

    # 4) Print final stock and total time
    print("\n===== RESULTS =====")
    print("Final stock:")
    for rsrc, amt in stock.items():
        print(f"  {rsrc}: {amt}")
    print(f"Total time consumed: {total_time}")
    print("===================")
