# IMPORT NECESSARY MODULES
# -----------------------------------------------------
import math
import os

try:
    from typing import List, Tuple, Callable
except ImportError:
    # Install the typing module first and then import it
    os.system("pip install typing")
    from typing import List, Tuple, Callable
# -----------------------------------------------------


# DEFINE CUSTOM VARIABLE TYPES
# -----------------------------------------------------

# Vectors are just list of numbers
Vector = List[float]

# Matrices are just lists of lists
Matrix = List[List[float]]

# -----------------------------------------------------


# FUNCTIONS FOR VECTORS
# -----------------------------------------------------


def add(v: Vector, w: Vector) -> Vector:
    """
    Performs an element-wise addition of two vectors

    Arguments:
    ----------
        - v   : Vector type; i.e. a list of the coordinates of the vector
        - w   : Vector type; i.e. a list of the coordinates of the vector

    Returns:
    --------
        - Vector type; i.e. a list of the coordinates of the new vector
    """

    assert len(v) == len(w), "vectors must have the same length"

    return [v_i + w_i for v_i, w_i in zip(v, w)]


def subtract(v: Vector, w: Vector) -> Vector:
    """
    Performs an element-wise subtraction of two vectors

    Arguments:
    ----------
        - v   : Vector type; i.e. a list of the coordinates of the vector
        - w   : Vector type; i.e. a list of the coordinates of the vector

    Returns:
    --------
        - Vector type; i.e. a list of the coordinates of the new vector
    """

    assert len(v) == len(w), "vectors must have the same length"

    return [v_i - w_i for v_i, w_i in zip(v, w)]


def vector_sum(vectors: List[Vector]) -> Vector:
    """
    Performs an element-wise addition of a list of vectors

    Arguments:
    ----------
        - vectors   : List[Vector] type; i.e. a list of Vector type objects

    Returns:
    --------
        - Vector type; i.e. a list of the coordinates of the new vector
    """

    # Check that the list of vectors is not empty
    assert vectors, "you didn't provide any vectors"

    # Check that all vectors in the list have the same length
    num_elements = len(vectors[0])
    assert all(
        len(v) == num_elements for v in vectors
    ), "One or more vectors have different length"

    # The i-th element is the sum of vector[i]
    return [sum(vector[i] for vector in vectors) for i in range(num_elements)]


def scalar_multiply(c: float, v: Vector) -> Vector:
    """
    Performs an element-wise multiplication of a vector with a scalar quantity

    Arguments:
    ----------
        - c   : Float type; the scalar quantity
        - v   : Vector type; i.e. a list of the coordinates of the vector

    Returns:
    --------
        - Vector type; i.e. a list of the coordinates of the new vector
    """
    return [c * v_i for v_i in v]


def vector_mean(vectors: List[Vector]) -> Vector:
    """
    Calculates the element-wise mean value of a list of vectors

    Arguments:
    ----------
        - vectors   : List type; a list of Vectors

    Returns:
    --------
        - Vector type; i.e. a list of the coordinates of the new vector
    """

    n = len(vectors)
    return scalar_multiply(1 / n, vector_sum(vectors))


def dot(v: Vector, w: Vector) -> float:
    """
    Calculates the inner product of two vectors

    Arguments:
    ----------
        - v   : Vector type; i.e. a list of the coordinates of the vector
        - w   : Vector type; i.e. a list of the coordinates of the vector

    Returns:
    --------
        - Float type; the inner product of the two vectors
    """

    assert len(v) == len(w), "vectors must have the same length"

    return sum(v_i * w_i for v_i, w_i in zip(v, w))


def sum_of_squares(v: Vector) -> float:
    """
    Calculates the sum of squares of a vector

    Arguments:
    ----------
        - v   : Vector type; i.e. a list of the coordinates of the vector

    Returns:
    --------
        - Float type; the sum of squares of the vector
    """
    return dot(v, v)


def magnitude(v: Vector) -> float:
    """
    Calculates the magnitude (length) of a vector

    Arguments:
    ----------
        - v   : Vector type; i.e. a list of the coordinates of the vector

    Returns:
    --------
        - Float type; the magnitude (length) of the vector
    """
    return math.sqrt(sum_of_squares(v))


def squared_distance(v: Vector, w: Vector) -> float:
    """
    Calculates the distance squared between two vectors

    Arguments:
    ----------
        - v   : Vector type; i.e. a list of the coordinates of the vector
        - w   : Vector type; i.e. a list of the coordinates of the vector

    Returns:
    --------
        - Float type; the distance squared between the two vectors
    """
    return sum_of_squares(subtract(v, w))


def distance(v: Vector, w: Vector) -> float:
    """
    Calculates the distance between two vectors

    Arguments:
    ----------
        - v   : Vector type; i.e. a list of the coordinates of the vector
        - w   : Vector type; i.e. a list of the coordinates of the vector

    Returns:
    --------
        - Float type; the distance between the two vectors
    """
    return magnitude(subtract(v, w))


# -----------------------------------------------------


### FUNCTIONS FOR MATRICES
# -----------------------------------------------------


def shape(A: Matrix) -> Tuple[int, int]:
    """
    Calculates the shape of a matrix

    Arguments:
    ----------
        - A   : Matrix type; i.e. a list of lists

    Returns:
    --------
        - Tuple type; it returns a tuple that contain two integers:
            i) the number of rows
            ii) the number of columns of the matrix

    """
    num_rows = len(A)
    num_cols = len(A[0]) if A else 0  # number of elements in first row

    return num_rows, num_cols


def get_row(A: Matrix, i: int) -> Vector:
    """
    Retrieves the i-th row of A as a vector

    Arguments:
    ----------
        - A   : Matrix type; i.e. a list of lists

    Returns:
    --------
        - Vector type; it returns the i-th row of A as a vector
    """
    return A[i]


def get_column(A: Matrix, j: int) -> Vector:
    """
    Retrieves the j-th column of A as a vector

    Arguments:
    ----------
        - A   : Matrix type; i.e. a list of lists

    Returns:
    --------
        - Vector type; it returns the j-th column of A as a vector
    """
    # j-th element of row A_i for every row A_i
    return [A_i[j] for A_i in A]


def make_matrix(
    num_rows: int, num_cols: int, entry_fn: Callable[[int, int], float]
) -> Matrix:
    """
    It returns a matrix with dimensions (num_rows x num_cols) where in the
    (i,j)-th position contains the value entry_fn(i,j)

    Arguments:
    ----------
        - num_rows   : Integer type; the number of rows
        - num_cols   : Integer type; the number of columns
        - entry_fn   : Callable type; a function that takes as input two integers i,j (row and column)
                                      and returns a float to be placed in position A[i,j]

    Returns:
    --------
        - Matrix type; i.e. a list of lists
    """

    # For a given i, make the list [entry_fn(i,0), ...] and make a list for every i
    return [[entry_fn(i, j) for j in range(num_cols)] for i in range(num_rows)]


def identity_matrix(n: int) -> Matrix:
    """
    It returns the identity matrix of dimensions (n x n)

     Arguments:
    ----------
        - n   : Integer type; the dimension of the squared matrix

    Returns:
    --------
        - Matrix type; i.e. a list of lists
    """
    return make_matrix(n, n, lambda i, j: 1 if i == j else 0)


# -----------------------------------------------------

if __name__ == "__main__":
	module_name = "linear_algebra_solutions"
	print("THIS IS THE {!r} MODULE".format(module_name))
