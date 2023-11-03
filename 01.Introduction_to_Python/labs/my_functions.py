import numpy as np


def my_sqrt(x, eps=1e-6, n_max=10000):
    """
        ARGUMENTS
        =========
            - x       : float or int
            - eps     : float or int. The accuracy of the estimation
            - n_max   : int. The maximum number of iterations to try
                        to converge to solution.
                        
        RETURNS
        =======
            float. The square root estimation for number x.
    """
    
    try:
        # Check validity of input
        if (x < 0) or (eps < 0) or (n_max < 0):
            raise ValueError

        # Define the boundaries for the search space
        a = 0
        b = max(x,1)

        # Divide and conquer: guess value in the middle of the search space
        guess = (a+b)/2

        # A counter to count the number of iterations required
        # for convergence
        n = 0
    
        while abs(guess**2 - x) > eps:
            
            if guess**2 < x:
                a = guess
                guess = (a+b)/2
                
            else:
                b = guess
                guess = (a+b)/2
            
            # Increment the counter
            n += 1
            
            if n > n_max:
                raise RuntimeError
                
    except ValueError:
        print("ERROR: x, eps, and n_max cannot be negative numbers")
        return None
                
    except RuntimeError:
        print("ERROR: could not convergence to solution")
        return None
    
    return guess