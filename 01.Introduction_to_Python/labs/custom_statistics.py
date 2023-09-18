# IMPORT NECESSARY MODULES
# -----------------------------------------------------
import math
import os
try:
    from typing import List
except ImportError:
    os.system("pip install typing")
    from typing import List
    
# You can replace this module with your own linear_algebra module
# Or skip it and re-implement the relevant functions here
from linear_algebra_solutions import sum_of_squares, dot
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

    # Replace the "pass" statement with appropriate code in order to return the correct quantity
    pass


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

    # Replace the "..." with appropriate code in order to return the correct quantity
    sorted_xs = ...

    # Replace the "..." with appropriate code in order to return the correct quantity
    if ...:
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

    # Replace the "pass" statement with appropriate code in order to return the correct quantity
    pass



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

    # Replace the "..." with appropriate code in order to return the correct quantity
    x_bar = ...
    return [... - x_bar for ... in ...]



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
    assert len(xs) >=2, "variance need at least 2 elements"

    # Replace the "..." with appropriate code in order to return the correct quantity
    # or re-write the entire function (in this case comment-out or overwrite this function)
    n = ...
    deviations = de_mean(xs)
    return sum_of_squares(...) / (...)



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
   
    # Replace the "pass" statement with appropriate code in order to return the correct quantity
    pass


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

    # Replace the "..." with appropriate code in order to return the correct quantity
    # or re-write the entire function (in this case comment-out or overwrite this function)
    return dot(..., ...) / (...)