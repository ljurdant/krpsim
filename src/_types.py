from typing import TypedDict


class Process(TypedDict):
    need: dict[str, int]
    result: dict[str, int]
    time: int
