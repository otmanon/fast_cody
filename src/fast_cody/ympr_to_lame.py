

'''
Converts from youngs modulus and poisson ratio to lame parameters.
'''
def ympr_to_lame(ym, pr):
    """
    Converts from youngs modulus and poisson ratio to lame parameters.

    Parameters
    ----------
    ym : float or float numpy array
        Youngs modulus
    pr : float or float numpy array
        Poisson ratio

    Returns
    -------
    mu : float or float numpy array
        First lame parameter
    lam : float or float numpy array
        Second lame parameter
    """
    mu = ym / (2*(1 + pr))
    lam = ym * pr / ((1 + pr)*(1 - 2*pr))
    return mu, lam