import tensorflow as tf
import monsetup
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D
import numpy

def getInitializedVariablesWithStandardDeviation( name, shape ):
    initializer = tf.initializers.TruncatedNormal(stddev=(2/shape[0])**0.5)
    return tf.Variable( initializer(shape=shape), name=name, dtype=tf.float32)

class EncoderDecoderModel( object ):
    def __init__( self, dimension ):
        self.Input_Layer_Size = (2**dimension)**2
        self.First_Hidden_Layer_Size = ((2**dimension)**2)/2
        self.Code_Layer_Size = 2**dimension
        self.Second_Hidden_Layer_Size = self.First_Hidden_Layer_Size
        self.Output_Layer_Size = self.Input_Layer_Size

        self.Weights_Input_Layer = getInitializedVariablesWithStandardDeviation([ self.Input_Layer_Size, self.First_Hidden_Layer_Size ], 'Input')
        self.Bias_Input_Layer =  getInitializedVariablesWithStandardDeviation(self.First_Hidden_Layer_Size, 'Input Bias')

        self.Weights_First_Hidden_Layer = getInitializedVariablesWithStandardDeviation([ self.First_Hidden_Layer_Size, self.Code_Layer_Size ], 'First Hidden')
        self.Bias_First_Hidden_Layer = getInitializedVariablesWithStandardDeviation(self.Code_Layer_Size, 'First Hidden Bias')

        self.Weights_Code_Layer = getInitializedVariablesWithStandardDeviation([ self.Code_Layer_Size, self.Second_Hidden_Layer_Size ], 'Code')
        self.Bias_Code_Layer = getInitializedVariablesWithStandardDeviation(self.Second_Hidden_Layer_Size, 'Code Bias')

        self.Weights_Second_Hidden_Layer = getInitializedVariablesWithStandardDeviation([ self.Second_Hidden_Layer_Size, self.Output_Layer_Size ], 'Second Hidden')
        self.Bias_Second_Hidden_Layer = getInitializedVariablesWithStandardDeviation(self.Output_Layer_Size, 'Code Bias')

    def __call__(self, input):
        first_layer_output = tf.nn.relu(self.Weights_Input_Layer * input + self.Bias_Input_Layer)
        second_layer_output = tf.nn.relu(self.Weights_First_Hidden_Layer * first_layer_output + self.Bias_First_Hidden_Layer)
        third_layer_output = tf.nn.relu(self.Weights_Code_Layer * second_layer_output + self.Bias_Code_Layer)
        output = tf.sign(self.Weights_Second_Hidden_Layer * third_layer_output + self.Bias_Second_Hidden_Layer)
        return output

def loss_function(predicted_y, target_y):
    return tf.reduce_mean(tf.square(predicted_y - target_y))


class EncoderDecoderKerasModel( Model ):
    def __init__(self,dimension):
        super(EncoderDecoderKerasModel, self).__init__()
        self.Input_Layer_Size = (2**dimension)**2
        self.First_Hidden_Layer_Size = ((2**dimension)**2)/2
        self.Code_Layer_Size = 2**dimension
        self.Second_Hidden_Layer_Size = self.First_Hidden_Layer_Size
        self.Output_Layer_Size = self.Input_Layer_Size
        self.d1 = Dense(self.First_Hidden_Layer_Size, activation='relu',input_dim=self.Input_Layer_Size, name="First")
        self.d2 = Dense(self.Code_Layer_Size, activation='relu',name="Second")
        self.d3 = Dense(self.Second_Hidden_Layer_Size, activation='relu',name="Third")
        self.d4 = Dense(self.Output_Layer_Size, activation='tanh',name="Fourth")

    def call(self, input):
        out1 = self.d1(input)
        out2 = self.d2(out1)
        out3 = self.d3(out2)
        return self.d4(out3)

DIMENSION = 4

model = EncoderDecoderKerasModel(DIMENSION)
optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

@tf.function
def train_step(inputs, outputs):
    with tf.GradientTape() as tape:
        predictions = model(inputs)
        loss = loss_function(outputs, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(outputs, predictions)

@tf.function
def test_step(inputs, outputs):
    predictions = model(inputs)
    t_loss = loss_function(outputs, predictions)

    test_loss(t_loss)
    test_accuracy(outputs, predictions)

train_ds = numpy.zeros( ( 2**(2**DIMENSION), (2**DIMENSION)**2 ), dtype=numpy.int8 )

for iterator in range ( 2**(2**DIMENSION) ):
    train_ds[iterator,:] = numpy.reshape(monsetup.qMatrixGenerator(iterator,DIMENSION), (1,(2**DIMENSION)**2) )

EPOCHS = 5

for epoch in range(EPOCHS):
  # Reset the metrics at the start of the next epoch
  train_loss.reset_states()
  train_accuracy.reset_states()
  test_loss.reset_states()
  test_accuracy.reset_states()

  for inputs in train_ds:
    train_step(inputs, inputs)

  for test_inputs in train_ds:
    test_step(test_inputs, test_inputs)

  template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
  print(template.format(epoch+1,
                        train_loss.result(),
                        train_accuracy.result()*100,
                        test_loss.result(),
                        test_accuracy.result()*100))