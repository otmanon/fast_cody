

'''
Converts from youngs modulus and poisson ratio to lame parameters.
'''
def ympr_to_lame(ym, pr):
    mu = ym / (2*(1 + pr))
    lam = ym * pr / ((1 + pr)*(1 - 2*pr))
    return mu, lam