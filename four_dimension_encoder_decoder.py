import tensorflow as tf
import monsetup
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Activation
import numpy

DIMENSION = 4

Input_Layer_Size = (2**DIMENSION)**2
First_Hidden_Layer_Size = ((2**DIMENSION)**2)/2
Code_Layer_Size = 2**DIMENSION
Second_Hidden_Layer_Size = First_Hidden_Layer_Size
Output_Layer_Size = Input_Layer_Size

train_ds = numpy.zeros( ( 2**(2**DIMENSION), (2**DIMENSION)**2), dtype=numpy.float32 )

for iterator in range ( 2**(2**DIMENSION) ):
    train_ds[iterator, :] = numpy.reshape(monsetup.q_matrix_generator(iterator, DIMENSION), (1, (2 ** DIMENSION) ** 2))

model = Sequential()
model.add( Dense(First_Hidden_Layer_Size, activation='relu', input_dim=Input_Layer_Size, name="First") )
model.add( Dense(Code_Layer_Size, activation='sigmoid', name="Second") )
model.add( Dense(Second_Hidden_Layer_Size, activation='relu', name="Third") )
model.add( Dense(Output_Layer_Size, activation='tanh', name="Fourth") )

model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['accuracy'])

model.fit(train_ds, train_ds, epochs=10, batch_size=32)

code_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer("Second").output)

#intermediate_output = code_layer_model.predict(numpy.reshape(train_ds[5, :], ( 1,(2**DIMENSION)**2 )) )
#output = model.predict(numpy.reshape(train_ds[5, :], ( 1,(2**DIMENSION)**2 )) )