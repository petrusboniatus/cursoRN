'''
Codigo perceptron multicapa modificado para soportar un numero N de capas y con una cantidad m de neuronaas
'''

from __future__ import print_function

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

import tensorflow as tf

# Parameters
learning_rate = 0.0001
training_epochs = 1000
batch_size = 100
display_step = 1

# Network Parameters
anchoCapa = 4096 # Podemos cambiar esto a voluntad
n_input = 784 # Numero de pixeles de los datos de entrada
n_classes = 10 # Numero de clases en las que queremos clasificar los datos de entrada

# tf Graph input, Con esto introducimos las entradas (x) y las salidas esperadas para ese modelo (y)
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])


# Funcion de creacion de la red
def multilayer_perceptron(entrada, TamanhoEntrada,anchura,TamanhoSalida,profundidad):

    '''

    :param entrada: Tensor de entrada, correspondiente a un lote de imagenes
    :param TamanhoEntrada: Numero de pixeles de cada imagen
    :param anchura: Numero de neuronas de las capas interiores
    :param TamanhoSalida: Numero de categorias en las que queremos clasficar la entrada = numero de neuronas de la capa de salida
    :param profundidad: Numero de capas de la red
    :return: Salida de la red sin incluir la funcion de activacion de la ultima capa
    '''


    capaSalida = 0 #O dios mio el sida
    for capa in range(0,profundidad):

        '''
            Cada iteracion del bucle crea una capa, como estamos multiplicando matrices(tensores de rango 2):

            M1xM2 = M3

            Numero de filas M3 = Numero filas M1
            Numero de columnas M3 = Numero columnas M2

        '''

        dimension = []
        if capa == 0:
            dimension.append(TamanhoEntrada)
        else:
            dimension.append(anchura)

        if capa == profundidad-1:
            dimension.append(TamanhoSalida)
        else:
            dimension.append(anchura)

        pesosNuevaCapa = tf.Variable(tf.random_normal(dimension))
        biasNuevaCapa = tf.Variable(tf.random_normal([dimension[1]]))

        if capa == 0:
            sumaPonderada = tf.add(tf.matmul(entrada, pesosNuevaCapa), biasNuevaCapa)
        else:
            sumaPonderada = tf.add(tf.matmul(capaSalida, pesosNuevaCapa), biasNuevaCapa)

        if capa == profundidad-1:
            capaSalida = sumaPonderada
        else:
            capaSalida = tf.nn.relu(sumaPonderada)

    return capaSalida



# LLamamos a la funcion de la red, creamos un perceptron con de 6 capas
pred = multilayer_perceptron(entrada=x,TamanhoEntrada=n_input,anchura=anchoCapa,TamanhoSalida=n_classes,profundidad=6)

# Definimos y optimizamos el error
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)



# Initializing the variables
init = tf.global_variables_initializer()

# Inicializamos las variables
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):


        # Entrenamos la red (ahora esta en modo entrenamiento)
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
            avg_cost += c / total_batch


        # Mostramos el resultado de la funcion de error para los pesos actuales
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))



        #Comprobamos el ratio de acierto de la red (ahora esta en modo funcionamiento normal)
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))

    print("Optimization Finished!")

