# IMPORT NECESSARY MODULES
# -----------------------------------------------------
import math
import os
import sys

try:
    from typing import List
except ImportError:
    os.system("pip install typing")
    from typing import List

try: 
    # You can replace this module with your own linear_algebra module
    # Or skip it and re-implement the relevant functions here
    from linear_algebra_solutions import sum_of_squares, dot
except ModuleNotFoundError:
    sys.exit("Module 'linear_algebra_solutions' not found!")


# -----------------------------------------------------


# FUNCTIONS FOR BASIC STATISTICS
# -----------------------------------------------------


def mean(xs: List[float]) -> float:
    """
    Calculates the mean of a collection of numbers

    Arguments:
    ----------
        - xs   : List type; a list of numbers

    Returns:
    --------
        Float type; the mean value
    """
    return sum(xs) / len(xs)


def median(xs: List[float]) -> float:
    """
    Calculates the median of a collection of numbers

    Arguments:
    ----------
        - xs   : List type; a list of numbers

    Returns:
    --------
        Float type; the median value
    """

    # To find the median value we first need to sort the collection
    # If the collection consists of an even number of elements, then
    # the median is the average (mean) of the two midpoints.
    # On the other hand, if the collection consists of an odd number
    # of elements, the median is simply the midpoint.
    sorted_xs = sorted(xs)

    if len(xs) % 2 == 0:
        high_midpoint = len(xs) // 2
        result = (sorted_xs[high_midpoint - 1] + sorted_xs[high_midpoint]) / 2
    else:
        result = sorted_xs[len(xs) // 2]

    return result


def data_range(xs: List[float]) -> float:
    """
    It calculates the range over which the data span;
    e.g. the data range of the sequence [4,2,3,1,10,8] is 10 - 1 = 9

    Arguments:
    ----------
        - xs   : List type; a list of numbers

    Returns:
    --------
        Float type; the data range
    """
    return max(xs) - min(xs)


def de_mean(xs: List[float]) -> List[float]:
    """
    Modifies the elements in xs by subtracting the mean value (so the new dataset has a mean value of 0.0)

    Arguments:
    ----------
        - xs   : List type; a list of numbers

    Returns:
    --------
        List type; a new dataset with a mean value of 0.0
    """
    x_bar = mean(xs)
    return [x - x_bar for x in xs]


def variance(xs: List[float]) -> float:
    """
    Calculates the variance

    Arguments:
    ----------
        - xs   : List type; a list of numbers

    Returns:
    --------
        Float type; the variance of the sample
    """
    assert len(xs) >= 2, "variance need at least 2 elements"

    n = len(xs)
    deviations = de_mean(xs)
    return sum_of_squares(deviations) / (n - 1)


def standard_deviation(xs: List[float]) -> float:
    """
    Calculates the standard deviation

    Arguments:
    ----------
        - xs   : List type; a list of numbers

    Returns:
    --------
        Float type; the standard deviation of the sample

    """
    return math.sqrt(variance(xs))


def covariance(xs: List[float], ys: List[float]) -> float:
    """
    Calculates the covariance between xs and ys

    Arguments:
    ----------
        - xs   : List type; a list of numbers
        - ys   : List type; a list of numbers

    Returns:
    --------
        Float type; the covariance between xs and ys
    """
    assert len(xs) == len(ys), "xs and ys must have the same length"

    return dot(de_mean(xs), de_mean(ys)) / (len(xs) - 1)

# -----------------------------------------------------------------------------------------------

if __name__ == "__main__":
	module_name = "custom_statistics_solutions"
	print("THIS IS THE {!r} MODULE".format(module_name))
