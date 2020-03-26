import numpy as np
from scipy.fftpack import fft
from scipy.fftpack import ifft


def sliding_dot_product(Q, T):
    """
    Calculates the sliding dot product between a input time series T
    and a query subsequence Q. This essentially calculates a correlation.

    Args:
        Q (np.array): a query subsequence of a time series
        T (np.array): an input time series

    Returns:
        A real-valued np.array containing the dot products of Q and T.
        The length of this array is 2 * len(T).
    """
    n = len(T)
    m = len(Q)

    Ta = np.concatenate((T, np.zeros(n)))
    Qra = np.concatenate((Q[::-1], np.zeros(2 * n - m)))

    return np.real(ifft(fft(Qra) * fft(Ta)))


def trailing_rolling_fun(fun, x, m):
    """
    Compute a trailing rolling function, probably a mean or standard deviation

    Args:
        fun: function to apply to the input array
        x (nd.array): the input array to which to apply a rolling function
        m (int): the size of the rolling window

    Returns:
        An np.array the same size as the input x that contains the rolling
        values for the applied function
    """
    if fun == np.mean:
        fun = np.nanmean
    elif fun == np.std:
        fun = np.nanstd
    x = np.pad(x.astype("float"), (m - 1, 0), mode='constant',
               constant_values=np.nan)
    return np.array([fun(x[i:(i + m)]) for i in range(len(x) - m + 1)])


def elementwise_min(P, I, D, idx, m, exclude_zone=None):
    """Calculates the elementwise minimum between the matrix profile and
       the full distance profile for a given subsequence starting at idx.

    Args:
        P (np.ndarray): the current matrix profile
        I (np.ndarray): the current matrix profile index
        D (np.ndarray): the current distance profile
        idx (int): The index of the current subsequence
        m (int): The subsequence length
        exclude_zone (int): If this is anything other than none, a region
                            of this size around the "trivial matches" will be
                            excluded

    Returns:
        A 2-tuple where the first element is the updated matrix profile and
        the second element is the updated matrix profile index
    """
    assert len(P) == len(D)

    P2 = np.minimum(P, D)
    I2 = I.copy()
    I2[np.where(D < P)] = idx
    if exclude_zone is not None:
        exclude_idx = range(max(0, idx - round(m / float(exclude_zone)) - 1),
                            min(len(P), idx + round(m / float(exclude_zone)) + 1))
        P2[exclude_idx] = P[exclude_idx]
        I2[exclude_idx] = I[exclude_idx]

    return P2, I2


def calc_distance_profile(QT, M, S, m, n, i, exclude=False):
    """
    Helper function to calculate the Euclidean distance from a dot product

    Args:
        QT (np.ndarray)
        M (np.ndarray): Pre-calculated mean for all subsequences in self-join case
        S (np.ndarray): Pre-calculated standard deviation for all subsequences in
                        self-join case
        m (int): Subsequence length
        n (int): Length of original time series
        i (int): Subsequence index of interest
        exclude (bool; default=False): Whether to use the exclusion region

    Returns:
        An np.ndarray of the distance profile (z-normalized Euclidean distance
        between the subsequence index of interest and the original timeseries)
    """
    muidx = i + m - 1
    dist = 2 * m * (1 - ((QT - m * M[muidx] * M[m - 1:n]) / (m * S[muidx] * S[m - 1:n])))
    if exclude:
        exclude_idx = range(max(0, i - round(m / 4.0) - 1),
                            min(len(dist), i + round(m / 4.0) + 1))
        dist[exclude_idx] = np.Inf

    return np.sqrt(np.abs(dist))


def mass(Q, T, M=None, S=None):
    """Muenn's Algorithm for Similarity Search

    Uses some FFT tricks to efficiently calculate the z-normalized distance
    profile between a query subsequence Q and a time series T.

    Args:
        Q (np.ndarray): A query time series (subsequence)
        T (np.ndarray): The original time series, np.ndarray
        M (np.ndarray): Pre-calculated mean for all subsequences in self-join case
        S (np.ndarray): Pre-calculated standard deviation for all subsequences in
                        self-join case
    Returns:
        An np.ndarray containing the Euclidean distances between Q and every
        subsequence of length len(Q) in T.

    Reference:
        Abdullah Mueen, Yan Zhu, Michael Yeh, Kaveh Kamgar, Krishnamurthy Viswanathan,
        Chetan Kumar Gupta and Eamonn Keogh (2015). "The Fastest Similarity Search
        Algorithm for Time Series Subsequences under Euclidean Distance"
        URL: http://www.cs.unm.edu/~mueen/FastestSimilaritySearch.html
    """
    QT = sliding_dot_product(Q, T)
    mu = np.mean(Q)
    sigma = np.std(Q)
    m = len(Q)
    n = len(T)

    if M is None:
        M = trailing_rolling_fun(np.nanmean, T, m)
        S = trailing_rolling_fun(np.nanstd, T, m)
    dist = 2 * m * (1 - ((QT[m - 1:n] - m * mu * M[m - 1:n]) / (m * sigma * S[m - 1:n])))

    # Distance can't be negative, but if S is really small, we may get a
    # numerical negative. So, let's fix that.
    dist = np.abs(dist)

    return np.sqrt(dist)


def stamp(TA, TB=None, m=4, n_iters=-1, exclusion_zone=4.0):
    """Scalable Time Series Anytime Matrix Profile (STAMP)
    This is an anytime algorithm to calculate the Matrix Profile of two input
    time series.

    Args:
        TA (np.array): An input time series
        TB (np.array, default=None): A second input time series. If None, performs
            a self-join.
        m (int, default=4): The subsequence window length. We're looking for
            features of this length.
        n_iters (int, default=-1): The number of iterations until you want the
            algorithm to stop (since it is anytime). If -1, just runs until
            completion.
        exclusion_zone (float, default=4.0): Divisor to calculate the
            exclusion zone. Zone runs from i - m/exclusion_zone to i + m/exclusion_zone

     Returns:
        A 2-tuple (np.array, np.array) where the first element is the matrix
        profile and the second element is the matrix profile index.

    Reference:
        Chin-Chia Michael Yeh, Yan Zhu, Liudmila Ulanova, Nurjahan Begum,
        Yifei Ding, Hoang Anh Dau, Diego Furtado Silva, Abdullah Mueen,
        Eamonn Keogh (2016). "Matrix Profile I: All Pairs Similarity Joins for
        Time Series: A Unifying View that Includes Motifs, Discords and Shapelets."
        IEEE ICDM 2016.
    """
    M = None
    S = None

    if TB is None:
        TB = TA.copy()
        M = trailing_rolling_fun(np.nanmean, TB, m)
        S = trailing_rolling_fun(np.nanstd, TB, m)

    nB = len(TB)
    nP = nB - m + 1

    P = np.empty(nP)
    P.fill(np.Inf)
    I = np.zeros(nP)

    idxs = range(nP)
    if n_iters != -1:
        idxs = np.random.permutation(idxs)

    i = 0
    for idx in idxs:
        i += 1
        D = mass(TB[idx:(idx + m)], TA, M, S)
        P, I  = elementwise_min(P, I, D, idx, m, exclusion_zone)
        if i == n_iters:
            return P, I
    return P, I


def stomp(T, m=4, exclusion_zone=4.0):
    """Scalable Time Series Ordered Matrix Profile (STOMP)
    This is an  algorithm to calculate the Matrix Profile of an input
    time series. At the moment, this only performs a self join (i.e.,
    doesn't take two different input time series).

    Also, it is considerably slower than STAMP (not entirely sure why, but
    probably because fft is way more optimized than a simple inner loop in here.
    In any case, this algorithm is meant to be parallelized.

    Args:
        T (np.array): An input time series
        m (int, default=4): The subsequence window length. We're looking for
            features of this length.
        exclusion_zone (float, default=4.0): Divisor to calculate the
            exclusion zone. Zone runs from i - m/exclusion_zone to i + m/exclusion_zone

     Returns:
        A 2-tuple (np.array, np.array) where the first element is the matrix
        profile and the second element is the matrix profile index.

    Reference:
    Zhu, Yan, Zachary Zimmerman, Nader Shakibay Senobari, Chin-Chia Michael Yeh,
    Gareth Funning, Abdullah Mueen, Philip Brisk, and Eamonn Keogh.
    "Matrix profile ii: Exploiting a novel algorithm and gpus to break the one
     hundred million barrier for time series motifs and joins."
     In Data Mining (ICDM), 2016 IEEE 16th International Conference on,
     pp. 739-748. IEEE, 2016.

    TODO:
       Parallelize!
    """
    nT = len(T)
    nP = nT - m + 1

    P = np.empty(nP)
    P.fill(np.Inf)
    I = np.zeros(nP)

    M = trailing_rolling_fun(np.nanmean, T, m)
    S = trailing_rolling_fun(np.nanstd, T, m)

    QT = sliding_dot_product(T[0:m], T)
    QT = QT[m - 1:nT]
    QT_first = np.copy(QT)
    D = calc_distance_profile(QT, M, S, m, nT, 0, True)

    P = np.copy(D)
    for i in range(1, nP):
        for j in range(nP - 1, 0, -1):
            QT[j] = QT[j - 1] - T[j - 1] * T[i - 1] + T[j + m - 1] * T[i + m - 1]
        QT[0] = QT_first[i]
        D = calc_distance_profile(QT, M, S, m, nT, i, True)
        P, I = elementwise_min(P, I, D, i, m, exclusion_zone)

    return P, I
