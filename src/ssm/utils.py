import numpy as np


def matrix_to_string(M, sep=","):
    """
    Convert a numpy.matrix to a simple multi-line string.
    Example format:
    '1 2 3\n4 5 6'
    """
    A = np.asarray(M)
    rows = [sep.join(str(x) for x in row) for row in A]
    return "\n".join(rows)


def string_to_matrix(s, dtype=float, sep=","):
    """
    Convert a multi-line string to a numpy.matrix.
    Expected format (space-separated, newline-separated):
        '1 2 3\\n4 5 6'

    Parameters
    ----------
    s : str
        String representation of the matrix.
    dtype : type
        Desired numeric type (e.g. float, int).

    Returns
    -------
    np.array
        Parsed matrix.
    """
    # Split into non-empty lines
    lines = [line.strip(sep) for line in s.strip().splitlines() if line.strip()]
    # Split each line into entries and convert to dtype
    rows = [[dtype(x) for x in line.split(sep)] for line in lines]
    return np.array(rows)


def array1d_to_string(arr):
    """
    Convert a 1D numpy array to a space-separated string.

    Parameters
    ----------
    arr : np.ndarray (1D)
        Array to convert.

    Returns
    -------
    str
        String like '1.0 2.5 3.0'.
    """
    arr = np.asarray(arr).ravel()  # ensure 1D
    return " ".join(str(x) for x in arr)


def string_to_array1d(s, dtype=float):
    """
    Convert a space-separated string to a 1D numpy array.

    Parameters
    ----------
    s : str
        String like '1.0 2.5 3.0'.
    dtype : type, optional
        Type of the output array elements (default: float).

    Returns
    -------
    np.ndarray
        1D array with values parsed from the string.
    """
    s = s.strip()
    if not s:
        return np.array([], dtype=dtype)
    return np.array([dtype(x) for x in s.split()], dtype=dtype)


