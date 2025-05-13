import multiprocessing as mp
from multiprocessing import Process
from typing import Callable, TypeVar

import numpy as np
import pandas as pd

T = TypeVar("T")


def parallelize_on_list(x: list[T], func: Callable) -> list[T]:
    """
    Apply the function to every list element exploiting multiprocessing.

    :param x: the input list
    :param func: the function to be applied
    :return: a list of results
    """
    pool = mp.Pool(processes=(mp.cpu_count() - 1))
    y = pool.map(func, x)
    pool.close()
    pool.join()

    return y


def parallelize_on_df(df: pd.DataFrame, func: Callable) -> pd.DataFrame:
    n_chunks = mp.cpu_count() - 1
    with mp.Pool(n_chunks) as pool:
        chunks = np.array_split(df, n_chunks)
        results = pool.map(func, chunks)
        return pd.concat(results, axis=1)


def run_in_parallel(functions: list[Callable[[], None]]):
    proc = []
    for i, fn in enumerate(functions):
        p = Process(target=fn)
        p.start()
        proc.append(p)
    for p in proc:
        p.join()
