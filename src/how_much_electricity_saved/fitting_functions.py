import numpy as np


def fitting_func_exp(
    x: float,
    a: float,
    b: float,
    c: float,
) -> float:
    return a * np.exp(b * x) + c


def fitting_func_poly3(
    x: float,
    a: float,
    b: float,
    c: float,
    d: float,
) -> float:
    return a * x**3 + b * x**2 + c * x + d


def fitting_func_poly4(
    x: float,
    a: float,
    b: float,
    c: float,
    d: float,
    e: float,
) -> float:
    return a * x**4 + b * x**3 + c * x**2 + d * x + e

def fitting_func_linear(
    x: float,
    a: float,
    b: float,
) -> float:
    return x * a + b
