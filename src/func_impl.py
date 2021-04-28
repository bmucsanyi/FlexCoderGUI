"""Module containing the supported functions."""

from typing import Callable

import config
from src.input_ import Input


class FullStateTupleError(Exception):
    """The state tuple is at maximum length, and as such cannot be extended
    any further. The maximum length is specified in config.py."""


def length(arr: list[int]) -> int:
    """Returns the length of the ``arr``."""
    return len(arr)


def sort(arr: list[int]) -> list[int]:
    """Returns a sorted copy of ``arr``."""
    return sorted(arr)


def take(i: int, arr: list[int]) -> list[int]:
    """Returns a copy of the first ``i`` elements of ``arr``."""
    return arr[:i]


def drop(i: int, arr: list[int]) -> list[int]:
    """Returns a copy of ``arr`` without the first ``i`` elements."""
    return arr[i:]


def reverse_func(arr: list[int]) -> list[int]:
    """Returns a reversed copy of ``arr``."""
    return list(reversed(arr))


def map_func(func: Callable, arr: list[int]) -> list[int]:
    """Returns ``arr`` with ``func`` applied to each of its elements."""
    return list(map(func, arr))


def filter_func(func: Callable, arr: list[int]) -> list[int]:
    """Returns the elements of ``arr`` that meet the constraint ``func``."""
    return list(filter(func, arr))


def zip_with(op: Callable, arr_1: list[int], arr_2: list[int]) -> list[int]:
    """Returns an array whose i-th element is the result of ``op`` applied to
    the i-th element of ``arr_1`` and ``arr_2``.
    """
    return [op(a, b) for (a, b) in zip(arr_1, arr_2)]


def copy_state_tuple(state_tuple: tuple[Input, ...], index: int) -> tuple[Input, ...]:
    """Returns a new state tuple that is ``state_tuple`` extended with a
    one-element tuple containing the ``index``-th element of ``state_tuple``.
    """
    if len(state_tuple) == config.STATE_TUPLE_LENGTH_HIB:
        raise FullStateTupleError("State tuple parameter is already at max length.")
    return state_tuple + (state_tuple[index],)
