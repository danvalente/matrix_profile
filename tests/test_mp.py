import numpy as np
from math import isclose

from matrix_profile.mp import trailing_rolling_fun
from matrix_profile.mp import elementwise_min
from matrix_profile.mp import sliding_dot_product
from matrix_profile.mp import mass
from matrix_profile.mp import stamp
from matrix_profile.mp import stomp


def test_trailing_rolling_mean():
    m = 3
    x = np.array([3, -1, 4, 2, 7, -1, 3, 8, 8, -5])
    expected = np.array([3, 1, 2, 1.66666667, 4.33333333, 2.66666667,
                         3, 3.33333333, 6.33333333, 3.66666667])
    observed = trailing_rolling_fun(np.mean, x, m)
    assert all(isclose(observed[i], expected[i], rel_tol=1e-6)
               for i in range(len(x)))


def test_trailing_rolling_std():
    m = 3
    x = np.array([3, -1, 4, 2, 7, -1, 3, 8, 8, -5])
    expected = np.array([0, 2, 2.1602469, 2.05480467, 2.05480467, 3.29983165,
                         3.26598632, 3.68178701, 2.3570226, 6.12825877])
    observed = trailing_rolling_fun(np.std, x, m)
    assert all(isclose(observed[i], expected[i], rel_tol=1e-6)
               for i in range(len(x)))


def test_elementwise_min_profile_no_self_join():
    P = np.array([1.1, 2, -3, 5, 4.3])
    D = np.array([2, 1, 1.1, 3.5, 2.3])
    I = np.zeros(len(P))

    exp_P, _ = ([1.1, 1, -3, 3.5, 2.3], [])
    obs_P, _ = elementwise_min(P, I, D, 2, 2)

    assert all(obs_P[i] == exp_P[i] for i in range(len(P)))


def test_elementwise_min_index_no_self_join():
    P = np.array([1.1, 2, -3, 5, 4.3])
    D = np.array([2, 1, 1.1, 3.5, 2.3])
    I = np.zeros(len(P))

    _, exp_I = ([], [0, 2, 0, 2, 2])
    _, obs_I = elementwise_min(P, I, D, 2, 2)

    assert all(obs_I[i] == exp_I[i] for i in range(len(P)))

    pass


def test_elementwise_min_profile_self_join():
    P = np.array([1.1, 2, -3, 5, 4.3])
    D = np.array([2, 1, 1.1, 3.5, 2.3])
    I = np.zeros(len(P))

    exp_P, _ = ([1.1, 2, -3, 5, 2.3], [])
    obs_P, _ = elementwise_min(P, I, D, 2, 2, exclude_zone=2)

    assert all(obs_P[i] == exp_P[i] for i in range(len(P)))


def test_elementwise_min_index_self_join():
    P = np.array([1.1, 2, -3, 5, 4.3])
    D = np.array([2, 1, 1.1, 3.5, 2.3])
    I = np.zeros(len(P))

    _, exp_I = ([], [0, 0, 0, 0, 2])
    _, obs_I = elementwise_min(P, I, D, 2, 2, exclude_zone=2)

    assert all(obs_I[i] == exp_I[i] for i in range(len(P)))

    pass


def test_sliding_dot_product():
    Q = np.array([2.3, 1.4, -3.4, 2, -1, 5, 4.4, 9.1, 0.2, -2])
    T = np.array([-1.7, 8, 5.4, -3.3, 0.2, 0.6, 1.2, -4, 9, 2.5])

    expected = np.array([3.4, -16.34, -24.67, 73.0, 74.78, 34.27, 0.62, 14.46,
                         -19.72, -49.19, 107.78, 49.47, 53.05, -7.28, 32.16,
                         -28.44, -5.1, 24.2, 5.75, 0])
    observed = sliding_dot_product(Q, T)

    assert all(isclose(observed[i], expected[i], rel_tol=1e-6)
               for i in range(len(Q)))


def test_mass():
    Q = np.array([2.3, 1.4, -3.4, 2, -1, 5])
    T = np.array([-1.7, 8, 5.4, -3.3, 0.2, 0.6, 1.2, -4, 9, 2.5])

    expected = np.array([4.08572285, 2.40922095, 3.89266047, 2.27998253, 3.91907364])
    observed = mass(Q, T)

    assert all(isclose(observed[i], expected[i], rel_tol=1e-8)
               for i in range(len(T) - len(Q)))


def test_stamp():
    z = np.array([0.03509291, 0.1362775, 0.03231132, 0.27346994, 0.35628827,
                  0.2600436, 0.78479652, 0.31610214, 0.0166754, 0.67752345,
                  0.01956649, 0.92677602, 0.63078703, 0.51840364, 0.29769206,
                  0.38203257, 0.18887555, 0.31532505, 0.14482764, 0.10253768,
                  0.6768223, 0.62705348, 0.45138077, 0.14295428, 0.89251321,
                  0.113622, 0.70267223, 0.86428502, 0.77964607, 0.99582663])

    n = len(z)
    m = 5

    expected_P = np.array([1.22072773, 0.89445648, 1.28439452, 1.55904077, 1.83217803,
                           1.53132101, 0.89855286, 0.95140651, 1.02087579,
                           1.3673001, 0.77812777, 1.36557673, 1.51326014,
                           2.23175888, 1.67122597, 1.36557673, 1.28439452,
                           1.2207277, 0.89445648, 0.77812777, 1.36409493,
                           0.89855286, 0.95140651, 1.02087579, 1.3673001,
                           1.33836327])
    expected_I = np. array([17, 18, 16, 21, 15, 20, 21, 22, 23, 24, 19, 15, 6, 20,
                            21, 11, 2, 0, 1, 10, 16, 6, 7, 8, 9, 2])

    observed_P, observed_I = stamp(z, m=m, exclusion_zone=2.0)
    assert all(isclose(observed_P[i], expected_P[i], rel_tol=1e-6)
               for i in range(n - m)) and all(isclose(observed_I[i],
                                                      expected_I[i],
                                                      rel_tol=1e-6) for i in range(n - m))


def test_stomp():
    z = np.array([0.03509291, 0.1362775, 0.03231132, 0.27346994, 0.35628827,
                  0.2600436, 0.78479652, 0.31610214, 0.0166754, 0.67752345,
                  0.01956649, 0.92677602, 0.63078703, 0.51840364, 0.29769206,
                  0.38203257, 0.18887555, 0.31532505, 0.14482764, 0.10253768,
                  0.6768223, 0.62705348, 0.45138077, 0.14295428, 0.89251321,
                  0.113622, 0.70267223, 0.86428502, 0.77964607, 0.99582663])

    n = len(z)
    m = 5

    expected_P = np.array([1.22072773, 0.89445648, 1.28439452, 1.55904077, 1.83217803,
                           1.53132101, 0.89855286, 0.95140651, 1.02087579,
                           1.3673001, 0.77812777, 1.36557673, 1.51326014,
                           2.23175888, 1.67122597, 1.36557673, 1.28439452,
                           1.2207277, 0.89445648, 0.77812777, 1.36409493,
                           0.89855286, 0.95140651, 1.02087579, 1.3673001,
                           1.33836327])
    expected_I = np. array([17, 18, 16, 21, 15, 20, 21, 22, 23, 24, 19, 15, 6, 20,
                            21, 11, 2, 0, 1, 10, 16, 6, 7, 8, 9, 2])

    observed_P, observed_I = stomp(z, m=m, exclusion_zone=2.0)
    assert all(isclose(observed_P[i], expected_P[i], rel_tol=1e-6)
               for i in range(n - m)) and all(isclose(observed_I[i],
                                                      expected_I[i],
                                                      rel_tol=1e-6) for i in range(n - m))


def test_stamp_equals_stomp():
    z = np.array([0.03509291, 0.1362775, 0.03231132, 0.27346994, 0.35628827,
                  0.2600436, 0.78479652, 0.31610214, 0.0166754, 0.67752345,
                  0.01956649, 0.92677602, 0.63078703, 0.51840364, 0.29769206,
                  0.38203257, 0.18887555, 0.31532505, 0.14482764, 0.10253768,
                  0.6768223, 0.62705348, 0.45138077, 0.14295428, 0.89251321,
                  0.113622, 0.70267223, 0.86428502, 0.77964607, 0.99582663])

    n = len(z)
    m = 5

    stamp_P, stamp_I = stamp(z, m=m, exclusion_zone=4.0)
    stomp_P, stomp_I = stomp(z, m=m, exclusion_zone=4.0)

    assert all(isclose(stamp_P[i], stomp_P[i], rel_tol=1e-6)
               for i in range(n - m)) and all(isclose(stamp_I[i],
                                                      stomp_I[i],
                                                      rel_tol=1e-6) for i in range(n - m))
