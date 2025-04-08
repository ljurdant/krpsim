from time import sleep
from time import time
import math
import sys
import curses
import os


def proper_round(n: float):
    if (n - round(n)) >= 0.5:
        n = math.ceil(n)
    else:
        n = round(n)
    return n


def ft_progress(lst):
    loading_constant = 25
    start = time()
    second = 0
    previous_message = ""
    for i in range(len(lst)):
        if i == 1:
            second = time()
        current = time()
        percent = proper_round(((i + 1) / len(lst) * 100))
        loading = proper_round(loading_constant * percent / 100)
        elapsed_time = current - start
        elapsed_time = "{:.2f}".format(elapsed_time)
        if i >= 1:
            ETA = (second - start) * (len(lst) - i - 1)
            ETA = "{:.2f}".format(ETA)
        else:
            ETA = 0
        message = (
            "ETA "
            + str(ETA)
            + "s ["
            + (3 - len(str(percent))) * " "
            + str(percent)
            + "%]["
            + loading * "="
            + ">"
            + (loading_constant - loading) * " "
            + "]"
            + str(i + 1)
            + "/"
            + str(len(lst))
            + " | elapsed time "
            + str(elapsed_time)
            + "s"
        )
        if i >= 1:
            moveup = "\033[A"
            print(moveup * int(len(message) / width + 1))
        if len(previous_message) > len(message):
            message += (len(previous_message) - len(message)) * " "
        previous_message = message
        width = os.get_terminal_size()[0]
        print(message, end="\r")
        yield lst[i]
