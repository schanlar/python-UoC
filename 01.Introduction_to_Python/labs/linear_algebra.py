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

def add(v:Vector, w:Vector) -> Vector:
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

    # Replace the "..." with appropriate code in order to return the correct quantity
    return [... + ... for ..., ... in zip(..., ...)]



def subtract(v:Vector, w:Vector) -> Vector:
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

    # Replace the "..." with appropriate code in order to return the correct quantity
    return [... - ... for ..., ... in zip(..., ...)]



def vector_sum(vectors:List[Vector]) -> Vector:
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
    assert vectors, "you didn\'t provide any vectors"

    # Check that all vectors in the list have the same length
    num_elements = len(vectors[0])
    assert all(len(v) == num_elements for v in vectors), "One or more vectors have different length"

    # The i-th element is the sum of vector[i]
    # Replace the "..." with appropriate code in order to return the correct quantity
    return [sum(... for ... in vectors) for ... in range(num_elements)]



def scalar_multiply(c:float, v:Vector) -> Vector:
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

    # Replace the "..." with appropriate code in order to return the correct quantity
    return ...



def vector_mean(vectors:List[Vector]) -> Vector:
    """
        Calculates the element-wise mean value of a list of vectors

        Arguments:
        ----------
            - vectors   : List type; a list of Vectors
    
        Returns:
        --------
            - Vector type; i.e. a list of the coordinates of the new vector
    """

    # Replace the "..." with appropriate code in order to return the correct quantity
    n = len(...)
    return scalar_multiply(1/n, vector_sum(...))



def dot(v:Vector, w:Vector) -> float:
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

    # Replace the "..." with appropriate code in order to return the correct quantity
    return sum(... * ... for ..., ... in zip(..., ...))



def sum_of_squares(v:Vector) -> float:
    """
        Calculates the sum of squares of a vector

        Arguments:
        ----------
            - v   : Vector type; i.e. a list of the coordinates of the vector

        Returns:
        --------
            - Float type; the sum of squares of the vector
    """

    # Replace the "..." with appropriate code in order to return the correct quantity
    return dot(..., ...)



def magnitude(v:Vector) -> float:
    """
        Calculates the magnitude (length) of a vector

        Arguments:
        ----------
            - v   : Vector type; i.e. a list of the coordinates of the vector

        Returns:
        --------
            - Float type; the magnitude (length) of the vector
    """

    # Replace the "..." with appropriate code in order to return the correct quantity
    return math.sqrt(sum_of_squares(...))



def squared_distance(v:Vector, w:Vector) -> float:
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

    # Replace the "..." with appropriate code in order to return the correct quantity
    return sum_of_squares(subtract(..., ...))



def distance(v:Vector, w:Vector) -> float:
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

    # Replace the "..." with appropriate code in order to return the correct quantity
    return magnitude(subtract(..., ...))


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

    # Replace the "..." with appropriate code in order to return the correct quantity
    num_rows = len(A)
    num_cols = len(...) if A else 0 # len(...) should return the number of elements in first row

    return ..., ...



def get_row(A:Matrix, i:int) -> Vector:
    """
        Retrieves the i-th row of A as a vector

        Arguments:
        ----------
            - A   : Matrix type; i.e. a list of lists

        Returns:
        --------
            - Vector type; it returns the i-th row of A as a vector
    """

    # Replace the "..." with appropriate code in order to return the correct quantity
    return ...



def get_column(A:Matrix, j:int) -> Vector:
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
    # Replace the "..." with appropriate code in order to return the correct quantity
    return ...



def make_matrix(num_rows:int,
                num_cols:int,
                entry_fn:Callable[[int, int], float]) -> Matrix:
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
    # Replace the "..." with appropriate code in order to return the correct quantity
    return [[entry_fn(..., ...) for ... in range(num_cols)] for ... in range(num_rows)]



def identity_matrix(n:int) -> Matrix:
    """
        It returns the identity matrix of dimensions (n x n)

         Arguments:
        ----------
            - n   : Integer type; the dimension of the squared matrix

        Returns:
        --------
            - Matrix type; i.e. a list of lists
    """
    # Replace the "..." with appropriate code in order to return the correct quantity
    return make_matrix(..., ..., lambda i,j: 1 if ... == ... else 0)


# -----------------------------------------------------