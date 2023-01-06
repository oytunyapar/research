import numpy


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = numpy.exp(x - numpy.max(x))
    return e_x / e_x.sum(axis=0)
