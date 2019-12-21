import tensorflow as tf

def getInitializedVariablesWithStandardDeviation( name, shape ):
    initializer = tf.initializers.TruncatedNormal(stddev=(2/shape[0])**0.5)
    return tf.Variable( initializer(shape=shape), name=name)
