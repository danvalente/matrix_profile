# The Matrix Profile

## Background
This library implements algorithms for calculating the
[matrix profile](http://www.cs.ucr.edu/~eamonn/MatrixProfile.html) between two
time series. The Matrix Profile is a meta time series which shows the minimum of the z-normalized Euclidean distance between
all pairs of subsequences of length _m_ in two time series. Think of it
this way: if you segment each time series into subsequences of length _m_, the
distance between each pair of subsequences can
populate a matrix. The Matrix Profile is calculated by taking the minimum of the
rows (or columns) of this matrix. When comparing two time series in this way, we
can think of it as a "join" of the time series, and so if we are calculating
the Matrix Profile of a time series with itself, we refer to this as a "self join."

Here's why it is important:
The minima of the Matrix Profile are the locations of repeated motifs in the
time series. Maxima are discords (anomalies). Shapelets can also be simply
calculated from the Matrix Profile. **Therefore, many problems in time series
analysis can be solved in some way using the Matrix Profile.**
Refer to the papers linked below for more details.

## Details
The algorithms implemented here are [STAMP](http://www.cs.ucr.edu/~eamonn/PID4481997_extend_Matrix%20Profile_I.pdf) and [STOMP](). STAMP is an
O(n^2* log(n)) algorithm, where n are the number of points in the time
series. STAMP, however, has  anytime implementation, and so is
approximately correct whenever the user chooses to abort the algorithm. STOMP
is, in principle, a faster algorithm (O(n^2)) though it has no
anytime implementation. **In practice, however, because STAMP is based on the
FFT (and thus _highly_ optimized), it tends
to be much faster than STOMP.** This library does not attempt to parallelize
either of these algorithms, and the original STOMP paper describes how that
algorithm can be implemented on GPUs, so there is potential for speed up there.
At this point STOMP is only implemented for self joins.

Also included is Muenn's Algorithm for Similarity Search (MASS), which quickly
calculates the z-normalized Euclidean distance between a query subsequence and a
time series (i.e., the Distance Profile for the subsequence).

## Usage

Usage is pretty straightforward. For self-joins (i.e., comparing a time series
to itself)

```
# assuming T is our time series
profile, index = stamp(T, m=200)
```

`index` contains the Matrix Profile Index, which is the index of the subsequence
that maps to the minimum of the current subsequence (so, its minimum distance
"sibling," if you will).

In order to run the algorithm as an anytime algorithm, you must declare the
number of iterations to run

```
profile, index = stamp(T, m=200, n_iters=1000)
```

If you want, you can change the exclusion zone to discard "trivial" matches.
The zone runs from i - m/exclusion_zone to i + m/exclusion_zone, where i is the
index of the subsequence being compared. This isn't necessary, though &mdash;
the default should be fine.

```
profile, index = stamp(T, m=200, exclusion_zone=10.0)
```

To join two time series is not much more difficult, though the index points
to the sequences *in the second time series*
```
profile, index = stamp(TA, TB, m=200)
```

Finally, if you have a query subsequence Q and you want to calculate its
z-normalized Euclidean distance to every subsequence of the same length in a
time series T:

```
distance = mass(Q, T)
```

See the examples in `../notebooks/example.ipynb` for example usage and output.
