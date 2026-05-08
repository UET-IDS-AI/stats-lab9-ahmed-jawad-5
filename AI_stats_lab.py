import numpy as np


# -------------------------------------------------
# Sparse 4 by 4 Joint PMF
# -------------------------------------------------

def joint_pmf(x, y):
    pmf_table = {
        (0, 0): 0.10, (0, 1): 0.05, (0, 2): 0.00, (0, 3): 0.00,
        (1, 0): 0.15, (1, 1): 0.20, (1, 2): 0.05, (1, 3): 0.00,
        (2, 0): 0.00, (2, 1): 0.10, (2, 2): 0.15, (2, 3): 0.05,
        (3, 0): 0.00, (3, 1): 0.00, (3, 2): 0.05, (3, 3): 0.10,
    }
    return pmf_table.get((x, y), 0.0)


# -------------------------------------------------
# Expectation (FIXED: use joint PMF directly)
# -------------------------------------------------

def expected_x():
    total = 0
    for x in range(4):
        for y in range(4):
            total += x * joint_pmf(x, y)
    return total


def expected_y():
    total = 0
    for x in range(4):
        for y in range(4):
            total += y * joint_pmf(x, y)
    return total


def expected_xy():
    total = 0
    for x in range(4):
        for y in range(4):
            total += x * y * joint_pmf(x, y)
    return total


# -------------------------------------------------
# Variance (FIXED consistency)
# -------------------------------------------------

def variance_x():
    ex = expected_x()

    ex2 = 0
    for x in range(4):
        for y in range(4):
            ex2 += (x ** 2) * joint_pmf(x, y)

    return ex2 - ex ** 2


def variance_y():
    ey = expected_y()

    ey2 = 0
    for x in range(4):
        for y in range(4):
            ey2 += (y ** 2) * joint_pmf(x, y)

    return ey2 - ey ** 2


# -------------------------------------------------
# Covariance / Correlation
# -------------------------------------------------

def covariance_xy():
    return expected_xy() - expected_x() * expected_y()


def correlation_xy():
    return covariance_xy() / np.sqrt(variance_x() * variance_y())


# -------------------------------------------------
# Var(X+Y)
# -------------------------------------------------

def variance_sum():
    exy = expected_x() + expected_y()

    exy2 = 0
    for x in range(4):
        for y in range(4):
            exy2 += ((x + y) ** 2) * joint_pmf(x, y)

    return exy2 - exy ** 2


# -------------------------------------------------
# Identity check (FIXED)
# -------------------------------------------------

def variance_identity_check():
    lhs = variance_sum()
    rhs = variance_x() + variance_y() + 2 * covariance_xy()
    return bool(np.isclose(lhs, rhs))
