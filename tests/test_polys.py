from zzopt.polys import EXPECTED_DEGREES, POLYS, degree


def test_starting_polynomial_degrees():
    for name, expected in EXPECTED_DEGREES.items():
        assert degree(POLYS[name]) == expected

