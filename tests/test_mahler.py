from zzopt.mahler import log_mahler_Q_z1mz
from zzopt.polys import POLYS, multiply


def test_log_mahler_product_additive_for_q1_q2():
    q1 = log_mahler_Q_z1mz(POLYS["Q1"], precision=60)
    q2 = log_mahler_Q_z1mz(POLYS["Q2"], precision=60)
    product = log_mahler_Q_z1mz(multiply(POLYS["Q1"], POLYS["Q2"]), precision=60)
    assert abs(product - (q1 + q2)) < 1e-10
