# -*- coding: utf-8 -*-

# Sample code to use string producer.

import tensorflow as tf
import numpy as np

#le pasa un datos y el tamaño maximo a representar, te devuelve un vector en el que la posicion corresponde con el dato
#represtar 
def one_hot(x, n):
    """
    :param x: label (int)
    :param n: number of bits
    :return: one hot code
    """
    o_h = np.zeros(n)  # llenamos todo a ceros y en la posicion de x metemos un 1
    o_h[x] = 1

    """ ESTO DA ERRORES POR TODOS LADOS 
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x), n))
    o_h[np.arange(len(x)), x] = 1
    """


    return o_h


num_classes = 3 #numero de clases
batch_size = 4


# --------------------------------------------------
#
#       DATA SOURCE
#
# --------------------------------------------------

def dataSource(paths, batch_size):
    min_after_dequeue = 10
    capacity = min_after_dequeue + 3 * batch_size

    example_batch_list = [] #numeros que te van a dar
    label_batch_list = [] #etiquetas que haces con one_hot

    #paths son todas las rutas de ficheros, enumerate enumera las listas del fichero.
    for i, p in enumerate(paths):
        #Apertura de ficheros
        filename = tf.train.match_filenames_once(p)
        filename_queue = tf.train.string_input_producer(filename, shuffle=False)
        reader = tf.WholeFileReader()
        #decoficacion de imagenes
        _, file_image = reader.read(filename_queue)
        image, label = tf.image.decode_jpeg(file_image), one_hot(int(i), num_classes)
        image = tf.image.resize_image_with_crop_or_pad(image, 80, 140)
        #image=tf.image.rgb_to_grayscale(image) #pasar fotos de 24 bits a 8 NO VALE PARA LAS DE 8
        image = tf.reshape(image, [80, 140, 1])
        image = tf.to_float(image) / 255. - 0.5
        example_batch, label_batch = tf.train.shuffle_batch([image, label], batch_size=batch_size, capacity=capacity,
                                                            min_after_dequeue=min_after_dequeue)
        example_batch_list.append(example_batch)
        label_batch_list.append(label_batch)

    #tf.contact para unir
    example_batch = tf.concat(values=example_batch_list, axis=0)
    label_batch = tf.concat(values=label_batch_list, axis=0)

    return example_batch, label_batch


# --------------------------------------------------
#
#       MODEL
#
# --------------------------------------------------

def myModel(X, reuse=False):
    with tf.variable_scope('ConvNet', reuse=reuse):
        #conv2d le pasas las imagenes (X) va a quitarle dos dimesiones a la matriz.
        o1 = tf.layers.conv2d(inputs=X, filters=32, kernel_size=3, activation=tf.nn.relu)
        #max_pooling2d cada 4 elementos coge el mas grande
        o2 = tf.layers.max_pooling2d(inputs=o1, pool_size=2, strides=2)
        o3 = tf.layers.conv2d(inputs=o2, filters=64, kernel_size=3, activation=tf.nn.relu)
        o4 = tf.layers.max_pooling2d(inputs=o3, pool_size=2, strides=2)

        h = tf.layers.dense(inputs=tf.reshape(o4, [batch_size * num_classes, 18 * 33 * 64]), units=5, activation=tf.nn.relu) ##modificada
        # units = numero de salidas (elementos que vas a usar) , softmax es para mucho elementos
        y = tf.layers.dense(inputs=h, units=num_classes, activation=tf.nn.softmax)
    return y



example_batch_train, label_batch_train = dataSource(["cos/entrenamiento/*.jpg", "infty/entrenamiento/*.jpg", "G/entrenamiento/*.jpg"], batch_size=batch_size)
example_batch_valid, label_batch_valid = dataSource(["cos/validacion/*.jpg", "infty/validacion/*.jpg", "G/validacion/*.jpg"], batch_size=batch_size)
example_batch_test, label_batch_test = dataSource(["cos/test/*.jpg", "infty/test/*.jpg", "G/test/*.jpg"], batch_size=batch_size)


#el reuse a true cuando usas el mismo modelo para que no se mezcle, la primera vez no necesitas true
example_batch_train_predicted = myModel(example_batch_train, reuse=False)
example_batch_valid_predicted = myModel(example_batch_valid, reuse=True)
example_batch_test_predicted = myModel(example_batch_test, reuse=True)

cost = tf.reduce_sum(tf.square(tf.cast(example_batch_train_predicted, tf.float64) - label_batch_train))
cost_valid = tf.reduce_sum(tf.square(tf.cast(example_batch_valid_predicted, tf.float64) - label_batch_valid))
cost_test = tf.reduce_sum(tf.square(tf.cast(example_batch_test_predicted, tf.float64) - label_batch_test))

##DEScenso del gradiente
# cost = tf.reduce_mean(-tf.reduce_sum(label_batch * tf.log(y), reduction_indices=[1]))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)
#learning_rate Cuanto mas pequeño aprende mas lento pero mas seguro

# --------------------------------------------------
#
#       TRAINING
#
# --------------------------------------------------

# Add ops to save and restore all the variables.

saver = tf.train.Saver()

with tf.Session() as sess:
    file_writer = tf.summary.FileWriter('./logs', sess.graph)

    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())

    # Start populating the filename queue.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    for _ in range(430):
        sess.run(optimizer)
        if _ % 20 == 0:
            print("Iter:", _, "---------------------------------------------")
            print(sess.run(label_batch_valid))
            print(sess.run(example_batch_valid_predicted))
            print("Error:", sess.run(cost_valid))

    save_path = saver.save(sess, "./tmp/model.ckpt")
    print("Model saved in file: %s" % save_path)

    coord.request_stop()
    coord.join(threads)
