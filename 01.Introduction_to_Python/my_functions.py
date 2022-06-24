import numpy as np

def my_sqrt(x, eps=1e-6):
    '''
    The function accepts as arguments the number from which 
    to estimate the square root and the required accuracy epsilon.
    '''
    
    # Check validity of input
    if (x < 0) or (eps < 0):
        raise ValueError("x and eps must not be negative numbers!")
        return None
    
    
    # Define the boundaries for the search space
    # If x > 1 then the boundaries will be [0,x]
    # If x < 1 then the boundaries will be [0,1]
    a = 0 
    b = max(x,1)

    middle = (a+b)/2
    
    # This serves as a flag variable
    n = 0
    
    while np.abs((middle**2 - x)) > eps and n <= 10000:
        
        if middle**2 < x:
            a = middle
            middle = (a+b)/2
        else:
            b = middle
            middle = (a+b)/2
        
        n += 1
        
    if n == 10000:
        print('Something went wrong with the loop!')
        return None
        
    else: 
        return middle