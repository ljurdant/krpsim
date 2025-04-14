import re


def parse(filepath):
    """
    Parses a krpsim configuration file with stricter error handling.

    Returns:
        stocks (dict[str, int]):
            Mapping from stock name to initial quantity.

        processes (dict[str, dict]):
            Each value is of the form:
                {
                    "need": dict[str, int],
                    "result": dict[str, int],
                    "time": int
                }

        optimize_targets (list[str]):
            A list of items to optimize (e.g. ["time", "happy_client"]).

    Raises:
        ValueError: If any line is malformed or if any quantity/delay
                    value is not an integer.
    """

    stocks = {}
    processes = {}
    optimize_targets = []

    # Regex to match a process line of the form:
    #   <process_name>:(needs):(results):delay
    # where (needs) = (res1:qty1;res2:qty2...)
    # and (results) = (res1:qty1;res2:qty2...)
    # process_name and delay are outside parentheses.
    process_pattern = re.compile(
        r"^([^:]+)"  # process_name (one or more non-":" characters)
        r":(?:\(([^)]*)\)|)"  # :(...needs...)
        r":(?:\(([^)]*)\)|)"  # :(...results...)
        r":(\d+)$"  # :delay
    )

    def parse_resource_pairs(pairs_str, line_context):
        """
        Given a string like "res1:qty1;res2:qty2;...", parse it into a dict.
        Raises ValueError if there's a bad format or non-integer quantity.
        line_context is only used for better error messages.
        """
        res_dict = {}
        if not pairs_str.strip():
            return res_dict  # no pairs; e.g. empty needs

        for pair in pairs_str.split(";"):
            pair = pair.strip()
            if not pair:
                continue
            if ":" not in pair:
                raise ValueError(
                    f"Malformed resource pair '{pair}' in line:\n  {line_context}"
                )
            resource, qty_str = pair.split(":", 1)
            resource = resource.strip()
            if not resource:
                raise ValueError(f"Empty resource name in line:\n  {line_context}")
            try:
                qty = int(qty_str.strip())
            except ValueError:
                raise ValueError(
                    f"Non-integer quantity '{qty_str}' for resource '{resource}' in line:\n  {line_context}"
                )
            if qty < 0:
                raise ValueError(
                    f"Negative quantity '{qty}' for resource '{resource}' in line:\n  {line_context}"
                )
            res_dict[resource] = qty

        return res_dict

    with open(filepath, "r") as f:
        for raw_line in f:
            line = raw_line.strip()

            # Skip empty lines or comment lines
            if not line or line.startswith("#"):
                continue

            # Check if it's the optimize line
            if line.startswith("optimize:"):
                # Example: optimize:(time;happy_client)
                if not (line.endswith(")") and "optimize:(" in line):
                    raise ValueError(f"Malformed optimize line:\n  {line}")

                # Extract what's inside the parentheses -> "time;happy_client"
                inside = line[
                    len("optimize:(") : -1
                ]  # remove "optimize:(" and trailing ")"
                # Split by ";" to get each target
                targets = [x.strip() for x in inside.split(";") if x.strip()]
                if not targets:
                    raise ValueError(f"No optimize targets found in line:\n  {line}")

                optimize_targets = targets
                continue

            # Check if it's a process line
            match = process_pattern.match(line)
            if match:
                proc_name = match.group(1).strip()
                needs_str = (match.group(2) or "").strip()
                results_str = (match.group(3) or "").strip()
                delay_str = match.group(4).strip()

                # Parse needs/results
                needs_dict = parse_resource_pairs(needs_str, line)
                results_dict = parse_resource_pairs(results_str, line)

                # Parse delay
                try:
                    delay = int(delay_str)
                except ValueError:
                    raise ValueError(
                        f"Delay is not an integer: '{delay_str}' in line:\n  {line}"
                    )
                if delay < 0:
                    raise ValueError(f"Negative delay '{delay}' in line:\n  {line}")

                # Store the process
                processes[proc_name] = {
                    "need": needs_dict,
                    "result": results_dict,
                    "time": delay,
                }

            else:
                # Otherwise, assume it's a stock line, e.g. "euro:10"
                # We'll parse strictly, requiring exactly one colon.
                parts = line.split(":")
                if len(parts) != 2:
                    raise ValueError(
                        f"Line not recognized as valid process or stock definition:\n  {line}"
                    )
                stock_name, qty_str = parts
                stock_name = stock_name.strip()
                if not stock_name:
                    raise ValueError(f"Empty stock name in line:\n  {line}")
                try:
                    qty = int(qty_str.strip())
                except ValueError:
                    raise ValueError(
                        f"Non-integer stock quantity '{qty_str}' for stock '{stock_name}' in line:\n  {line}"
                    )
                if qty < 0:
                    raise ValueError(
                        f"Negative stock quantity '{qty}' for stock '{stock_name}' in line:\n  {line}"
                    )
                stocks[stock_name] = qty

    if not stocks:
        raise ValueError("No stocks defined in the file.")
    if not processes:
        raise ValueError("No processes defined in the file.")
    if not optimize_targets:
        raise ValueError("No optimize targets defined in the file.")
    return stocks, processes, optimize_targets
