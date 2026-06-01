from zzopt.integral import integral_logplus
from zzopt.mahler import log_mahler_product
from zzopt.polys import POLYS, degree


def test_doche_baseline_normalization():
    q_factors = [(POLYS["Q1"], 1), (POLYS["Q2"], 1)]
    numerator = [POLYS[name] for name in ["P1", "P2", "P4", "P6", "P8"]]
    q = [13.1, 10.6, 3.2, 1.15, 0.24]
    deg_q = sum(degree(poly) * mult for poly, mult in q_factors)
    log_m = float(log_mahler_product(q_factors, fast=True))
    integral = integral_logplus(q_factors, numerator, q, grid_size=16384)
    bound = (log_m + integral) / deg_q
    assert 0.2535 <= bound <= 0.2555

