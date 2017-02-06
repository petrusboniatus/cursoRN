
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
import numpy as np


FLAGS = tf.app.flags.FLAGS


def read_images_from_disk(input_queue):
    label = input_queue[1]
    file_contents = tf.read_file(input_queue[0])
    example = tf.image.decode_png(file_contents, channels=1)
    procesada = tf.image.resize_images(example, [28, 28])
    return procesada, label

def transformacionVariables(listaFicheros,fmtCategorias):
    images = ops.convert_to_tensor(listaFicheros,dtype=dtypes.string)
    labels= ops.convert_to_tensor(fmtCategorias,dtype=dtypes.float64)
    input_queue = tf.train.slice_input_producer([images, labels], shuffle=True)
    image, label = read_images_from_disk(input_queue)
    image_batch, label_batch = tf.train.shuffle_batch([image, label],
                                                      batch_size=50,
                                                      capacity=50*1000,
                                                      min_after_dequeue=4)
    return image_batch, label_batch

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def modelo(image_batch, label_batch, image_test,label_test, dropout):

    y_ = label_batch

    # capa de circunvalación y bias
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])

    # Realizamos las operaciones con la primera capa
    x_image = tf.reshape(image_batch, [-1, 28, 28, 1])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    # Realizamos  la segunda de capa circunvalación y sus operaciones
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    # Creamos una capa completamente conectada
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # Dropout, consultar
    h_fc1_drop = tf.nn.dropout(h_fc1, dropout)

    # añadimos la ultima capa
    W_fc2 = weight_variable([1024, 4])
    b_fc2 = bias_variable([4])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    #construimos el test
    x_image_test = tf.reshape(image_test, [-1, 28, 28, 1])
    h_conv1_test =  tf.nn.relu(conv2d(x_image_test, W_conv1) + b_conv1)
    h_pool1_test = max_pool_2x2(h_conv1_test)
    h_conv2_test = tf.nn.relu(conv2d(h_pool1_test, W_conv2) + b_conv2)
    h_pool2_test = max_pool_2x2(h_conv2_test)
    h_pool2_flat_test = tf.reshape(h_pool2_test, [-1, 7 * 7 * 64])
    h_fc1_test = tf.nn.relu(tf.matmul(h_pool2_flat_test, W_fc1) + b_fc1)
    h_fc1_drop_test = tf.nn.dropout(h_fc1_test, dropout)
    y_conv_test = tf.matmul(h_fc1_drop_test, W_fc2) + b_fc2

    correct_prediction = tf.equal(tf.argmax(y_conv_test, 1), tf.argmax(label_test, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return train_step,accuracy

#creamos la lista de categorias y las transformamos
listaCategorias = np.loadtxt("resultadoColumnas.txt")
fmtCategorias = np.array([]) 

for i in listaCategorias:
    anadir =  [0,0,0,0]
    anadir[i.astype(int)-1]=1
    fmtCategorias = np.append(fmtCategorias,anadir)
fmtCategorias = fmtCategorias.reshape(-1,4)

#creamos la lista de ficheros
#cargamos las imagenes
listaFicheros = []
for i in range(190):
    fichero = "img/im-"+str(i)+".png"
    listaFicheros.append(fichero)

#Modelo de entrenamiento
image_batch, label_batch = transformacionVariables(listaFicheros[:130], fmtCategorias[:130])
#Modelo de prueba
image_test, label_test = transformacionVariables(listaFicheros[130:],fmtCategorias[130:])

paso_entreno, precision = modelo(image_batch,label_batch,image_test,label_test,1)


sess = tf.Session()
sess.run(tf.global_variables_initializer())

tf.train.start_queue_runners(sess=sess)

for i in range(1000):
    sess.run([paso_entreno])
    if i % 3 == 0:
        print(sess.run([precision]))