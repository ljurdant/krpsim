import re


def parse(filepath):
    """
    Parses a krpsim configuration file.

    Returns:
        stocks (dict[str, int]):
            Mapping from stock name to initial quantity.

        processes (list[dict]):
            Each element is a dictionary of the form:
                {
                    "name": str,
                    "needs": dict[str, int],
                    "results": dict[str, int],
                    "delay": int
                }

        optimize_targets (list[str]):
            A list of items to optimize (e.g. ["time", "happy_client"]).

    Notes:
        - Ignores lines starting with '#'.
        - For lines describing a process, we assume the format:
            name:(need1:qty1;need2:qty2...):(result1:qty1;result2:qty2...):delay
        - For the 'optimize' line, we assume the format:
            optimize:(target1;target2;...)
          e.g. optimize:(time;happy_client)
    """

    stocks = {}
    processes = {}
    optimize_targets = []

    # Regex to match a process line of the form:
    # <process_name>:(needs):(results):delay
    # where (needs) = (res1:qty1;res2:qty2...)
    # and (results) = (res1:qty1;res2:qty2...)
    # process_name and delay are outside parentheses.
    process_pattern = re.compile(
        r"^([^:]+)"  # process_name (one or more non-":" characters)
        r":\(([^)]*)\)"  # :(...needs...)
        r":\(([^)]*)\)"  # :(...results...)
        r":(\d+)$"  # :delay
    )

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()

            # Skip empty or comment lines
            if not line or line.startswith("#"):
                continue

            # Check if this is the optimize line
            if line.startswith("optimize:"):
                # Example: optimize:(time;happy_client)
                # Extract what's inside the parentheses -> "time;happy_client"
                inside = line[
                    len("optimize:(") : -1
                ]  # remove "optimize:(" and trailing ")"
                # Split by ";" to get each target
                optimize_targets = inside.split(";")
                optimize_targets = [x.strip() for x in optimize_targets if x.strip()]
                continue

            # Check if it's a process line
            match = process_pattern.match(line)
            if match:
                # It's a process line
                proc_name = match.group(1).strip()
                needs_str = match.group(2).strip()
                results_str = match.group(3).strip()
                delay_str = match.group(4).strip()

                # Parse needs
                needs_dict = {}
                if needs_str:
                    for part in needs_str.split(";"):
                        part = part.strip()
                        if not part:
                            continue
                        resource, qty_str = part.split(":")
                        needs_dict[resource.strip()] = int(qty_str)

                # Parse results
                results_dict = {}
                if results_str:
                    for part in results_str.split(";"):
                        part = part.strip()
                        if not part:
                            continue
                        resource, qty_str = part.split(":")
                        results_dict[resource.strip()] = int(qty_str)

                # Convert delay to int
                delay = int(delay_str)

                processes[proc_name] = {
                    "need": needs_dict,
                    "result": results_dict,
                    "time": delay,
                }
                processes["stop"] = {
                    "need": {},
                    "result": {},
                    "time": 0,
                }
            else:
                # Otherwise, assume it's a stock line, e.g. "euro:10"
                if ":" in line:
                    stock_name, qty_str = line.split(":", 1)
                    stocks[stock_name.strip()] = int(qty_str.strip())

    return stocks, processes, optimize_targets
