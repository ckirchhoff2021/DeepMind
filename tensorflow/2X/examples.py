import numpy as np
import pandas as pd
import tensorflow as tf
from common_path import *
import matplotlib.pyplot as plt
import tensorflow.keras as keras

import random
import seaborn as sns
import kerastuner as kt
from PIL import Image


def test_001():
    mnist = tf.keras.datasets.mnist
    (x1, y1), (x2, y2) = mnist.load_data()
    x1, x2 = x1 / 255.0, x2 / 255.0
    def build_model():
        model = tf.keras.models.Sequential(
            [
                tf.keras.layers.Flatten(input_shape=(28,28)),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(10, activation='softmax')
            ]
        )
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    model = build_model()
    # model.summary()

    def callback_save():
        save_callback = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(output_path, 'keras/test001.ckpt'), save_weights_only=True, verbose=1)
        model.fit(x1, y1, epochs=5, validation_data=(x2, y2), callbacks=[save_callback])

    def model_save():
        model.fit(x1, y1, epochs=5, validation_data=(x2, y2))
        model.save_weights(os.path.join(output_path, 'keras/test001.ckpt'))

    def checkpoint_load():
        model.load_weights(os.path.join(output_path, 'keras/test001.ckpt'))
        loss, acc = model.evaluate(x2, y2, verbose=2)
        print("accuracy: {:5.2f}%".format(100 * acc))

    def pb_save():
        model.fit(x1, y1, epochs=5, validation_data=(x2, y2))
        model.save(os.path.join(output_path, 'keras'))

    def pb_load():
        net = tf.keras.models.load_model(os.path.join(output_path, 'keras'))
        net.summary()
        loss, acc = net.evaluate(x2, y2, verbose=2)
        print("accuracy: {:5.2f}%".format(100 * acc))

    def h5_save():
        model.fit(x1, y1, epochs=5, validation_data=(x2, y2))
        model.save(os.path.join(output_path, 'keras_test001.h5'))

    def h5_load():
        net = tf.keras.models.load_model(os.path.join(output_path, 'keras_test001.h5'))
        net.summary()
        loss, acc = net.evaluate(x2, y2, verbose=2)
        print("accuracy: {:5.2f}%".format(100 * acc))

    # model_save()
    # checkpoint_load()
    # pb_save()
    # pb_load()
    # h5_save()
    h5_load()


def test_002():
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (x1, y1), (x2, y2) = fashion_mnist.load_data()
    x1, x2 = x1/255.0, x2/255.0
    class_names = [
        'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
    ]

    print(x1.shape)
    print(y1)

    # plt.figure()
    # plt.imshow(x1[8])
    # plt.colorbar()
    # plt.grid(False)
    # plt.show()

    # plt.figure(figsize=(10, 10))
    # for i in range(25):
    #     plt.subplot(5,5,i+1)
    #     plt.xticks([])
    #     plt.yticks([])
    #     plt.imshow(x1[i], cmap=plt.cm.binary)
    #     plt.xlabel(class_names[y1[i]])
    # plt.show()

    net = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28,28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)
    ])

    net.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    net.fit(x1, y1, epochs=10)
    _, accuracy = net.evaluate(x2, y2, verbose=2)
    print('ACC = ', accuracy)

    probabilities = tf.keras.Sequential([
        net,
        tf.keras.layers.Softmax()
    ])
    predicts = probabilities(x2)
    yp = predicts[0]
    print('yp: ', yp)
    ylabel = np.argmax(yp)
    yreal = y2[0]
    print('ylabel: %d, yreal: %d， yprob: %f' % (ylabel, yreal, yp[ylabel]))


def test_003():
    dataset_path = keras.utils.get_file(
        "auto-mpg.data", "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")

    column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                    'Acceleration', 'Model Year', 'Origin']
    raw_dataset = pd.read_csv(dataset_path, names=column_names,
                              na_values="?", comment='\t',
                              sep=" ", skipinitialspace=True)

    dataset = raw_dataset.copy()
    dataset = dataset.dropna()
    origin = dataset.pop('Origin')
    dataset['USA'] = (origin == 1) * 1.0
    dataset['Europe'] = (origin == 2) * 1.0
    dataset['Japan'] = (origin == 3) * 1.0


    train_dataset = dataset.sample(frac=0.8, random_state=0)
    test_dataset = dataset.drop(train_dataset.index)

    sns.pairplot(train_dataset[["MPG", "Cylinders", "Displacement", "Weight"]], diag_kind="kde")
    # plt.show()
    train_stats = train_dataset.describe()
    train_stats.pop("MPG")
    train_stats = train_stats.transpose()

    # print(train_dataset.tail())
    # print(train_stats)

    train_labels = train_dataset.pop('MPG')
    test_labels = test_dataset.pop('MPG')

    def norm(x):
        return (x - train_stats['mean']) / train_stats['std']

    normed_train_data = norm(train_dataset)
    normed_test_data = norm(test_dataset)
    
    # print(normed_train_data.tail())

    def build_model():
        model = keras.Sequential([
            keras.layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(1)
        ])

        optimizer = tf.keras.optimizers.RMSprop(0.001)

        model.compile(loss='mse',
                      optimizer=optimizer,
                      metrics=['mae', 'mse'])
        return model

    model = build_model()
    model.summary()
    example_batch = normed_train_data[:10]
    example_result = model.predict(example_batch)
    print(example_result)

    class PrintDot(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs):
            if epoch % 100 == 0: print('')
            print('.', end='')

    EPOCHS = 1000
    history = model.fit(
        normed_train_data, train_labels,
        epochs=EPOCHS, validation_split=0.2, verbose=0,
        callbacks=[PrintDot()])

    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    hist.tail()

    def plot_history(history):
        hist = pd.DataFrame(history.history)
        hist['epoch'] = history.epoch

        plt.figure()
        plt.xlabel('Epoch')
        plt.ylabel('Mean Abs Error [MPG]')
        plt.plot(hist['epoch'], hist['mae'],
                 label='Train Error')
        plt.plot(hist['epoch'], hist['val_mae'],
                 label='Val Error')
        plt.ylim([0, 5])
        plt.legend()

        plt.figure()
        plt.xlabel('Epoch')
        plt.ylabel('Mean Square Error [$MPG^2$]')
        plt.plot(hist['epoch'], hist['mse'],
                 label='Train Error')
        plt.plot(hist['epoch'], hist['val_mse'],
                 label='Val Error')
        plt.ylim([0, 20])
        plt.legend()
        plt.show()

    plot_history(history)


def kerastune_test():
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (x1, y1), (x2, y2) = fashion_mnist.load_data()
    x1, x2 = x1 / 255.0, x2 / 255.0

    def model_builder(hp):
        model = keras.Sequential()
        model.add(keras.layers.Flatten(input_shape=(28, 28)))

        hp_units = hp.Int('units', min_value=32, max_value=512, step=32)
        model.add(keras.layers.Dense(units=hp_units, activation='relu'))
        model.add(keras.layers.Dense(10))
        hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

        model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                      loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

        return model

    tuner = kt.Hyperband(model_builder,
                         objective='val_accuracy',
                         max_epochs=10,
                         factor=3,
                         directory='my_dir',
                         project_name='intro_to_kt')

    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    tuner.search(x1, y1, epochs=50, validation_split=0.2, callbacks=[stop_early])
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    print("""
    The hyperparameter search is complete. The optimal number of units in the first densely-connected
    layer is {best_hps.get('units')} and the optimal learning rate for the optimizer
    is {best_hps.get('learning_rate')}.
    """)

    model = tuner.hypermodel.build(best_hps)
    history = model.fit(img_train, label_train, epochs=50, validation_split=0.2)

    val_acc_per_epoch = history.history['val_accuracy']
    best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
    print('Best epoch: %d' % (best_epoch,))

    hypermodel = tuner.hypermodel.build(best_hps)

    # Retrain the model
    hypermodel.fit(x2, y2, epochs=best_epoch)


import time
def timeit(ds, batch_size, steps):
  overall_start = time.time()
  # 在开始计时之前
  # 取得单个 batch 来填充 pipeline（管道）（填充随机缓冲区）
  it = iter(ds.take(steps+1))
  next(it)

  start = time.time()
  for i,(images,labels) in enumerate(it):
    if i%10 == 0:
      print('.',end='')
  print()
  end = time.time()

  duration = end-start
  print("{} batches: {} s".format(steps, duration))
  print("{:0.5f} Images/s".format(batch_size*steps/duration))
  print("Total time: {}s".format(end-overall_start))

def test004():
    # import pathlib
    # data_root_orig = tf.keras.utils.get_file(
    #     origin='https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
    #     fname='flower_photos', untar=True)
    # data_root = pathlib.Path(data_root_orig)

    data_root = "/Users/chenxiang/.keras/datasets/flower_photos"
    image_folders = os.listdir(data_root)
    class_name = list()
    data_list = list()

    AUTOTUNE = tf.data.experimental.AUTOTUNE

    for folder in image_folders:
        if not os.path.isdir(os.path.join(data_root, folder)):
            continue
        class_name.append(folder)
        image_list = os.listdir(os.path.join(data_root, folder))
        for image in image_list:
            if not image.endswith('.jpg'):
                continue
            data_list.append([os.path.join(os.path.join(data_root, folder), image), folder])
    
    random.shuffle(data_list)
    print(class_name)

    datas = np.array(data_list)
    image_files = datas[:,0]
    image_labels = [class_name.index(x) for x in datas[:,1]]

    def preprocess_image(image):
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [192, 192])
        image /= 255.0
        image = image * 2.0 - 1.0
        return image

    def load_and_preprocess_image(path):
        image = tf.io.read_file(path)
        return preprocess_image(image)

    ds = tf.data.Dataset.from_tensor_slices((image_files, image_labels))
    def preprocess(image_file, image_label):
        return load_and_preprocess_image(image_file), image_label

    batch_size = 32
    image_label_ds = ds.map(preprocess)
    ds = image_label_ds.shuffle(buffer_size=len(data_list))
    ds = ds.repeat()
    ds = ds.batch(batch_size=batch_size)
    ds = ds.prefetch(buffer_size=AUTOTUNE)

    mobile_net = tf.keras.applications.MobileNetV2(input_shape=(192, 192, 3), include_top=False)
    mobile_net.trainable = False
    model = tf.keras.Sequential([
        mobile_net,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(len(class_name), activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss='sparse_categorical_crossentropy',
                  metrics=["accuracy"])

    model.summary()    
    # model.fit(ds, epochs=10, steps_per_epoch=3)
    steps_per_epoch = tf.math.ceil(len(data_list) / batch_size).numpy()

    '''
    ds2 = image_label_ds.cache(filename='cache.tf-data')
    ds2 = ds2.apply( tf.data.experimental.shuffle_and_repeat(buffer_size=len(data_list)))
    ds2 = ds2.batch(batch_size).prefetch(buffer_size=AUTOTUNE)
    timeit(ds2, batch_size, 2*steps_per_epoch + 1)
    '''

    '''
    image_ds = tf.data.Dataset.from_tensor_slices(image_files).map(load_and_preprocess_image)
    image_ds = image_ds.map(tf.io.serialize_tensor)
    tfrec = tf.data.experimental.TFRecordWriter('images.tfrec')
    tfrec.write(image_ds)
    '''

    def parse(x):
        result = tf.io.parse_tensor(x, out_type=tf.float32)
        result = tf.reshape(result, [192, 192, 3])
        return result

    image_ds = tf.data.TFRecordDataset('images.tfrec').map(parse, num_parallel_calls=AUTOTUNE)
    label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(image_labels, tf.int64))
    ds3 = tf.data.Dataset.zip((image_ds, label_ds))
    ds3 = ds3.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=len(image_list)))
    ds3 = ds3.batch(batch_size).prefetch(AUTOTUNE)
    timeit(ds3, batch_size, 2 * steps_per_epoch + 1)


def focal_loss_test():
    def loss(logits, labels, alpha, gamma):
        predictions = tf.nn.sigmoid(logits)
        zeros = tf.zeros_like(predictions, dtype=predictions.dtype)
        pos = tf.where(labels > zeros, labels-predictions, zeros)
        neg = tf.where(labels > zeros, zeros, predictions)
        fl_loss = -alpha * (pos ** gamma) * tf.math.log(predictions) - (1.0 - alpha) * (neg ** gamma) * tf.math.log(1.0 - predictions)
        fl_loss = tf.reduce_mean(fl_loss)
        return fl_loss

    logit = tf.random.normal([10,1])
    labels = tf.transpose(tf.constant([[1,0,1,0,1,1,1,0,0,1]],dtype=tf.float32))
    alpha = 0.25
    gamma = 2

    value = loss(logit, labels, alpha, gamma)
    print(value)


def logical_operator():
    x1 = tf.constant(['a', 'b'])
    x2 = tf.constant(['b', 'a'])

    x3 = tf.logical_and(x1=='a', x2=='b')
    x4 = tf.logical_or(x1=='a', x2=='b')
    print(x3)
    print(x4)

    x5 = tf.zeros_like(x1, dtype=tf.bool)
    print(x1=='a')
    print(x2=='a')
    print(x5)

    x6 = tf.where(x1=='e', x1, '0')
    print(x6)

    x7 = tf.cast(x3, tf.float32)
    print(x7)

    ids = tf.SparseTensor(indices=[[0, 1],
                                   [0, 3],
                                   [1, 2],
                                   [1, 3]],
                          values=['Y', 'N', 'Y', 'N'],
                          dense_shape=[2, 4])

    print(tf.compat.v1.sparse_tensor_to_dense(ids))
    x8 = tf.constant([12, 5, 6, 1, 17])
    y = tf.logical_and(x8>=6, x8 <=12)
    print('y:', y)


def loss_check():
    new_tags = tf.constant(['Y', 'Y', 'N', 'N', 'Y', 'N', 'N', 'N'])
    inactive_tags = tf.constant(['Y', 'Y', 'Y', 'N', 'Y', 'Y', 'Y', 'N'])
    new_coords = tf.where(new_tags == 'Y', True, False)
    inactive_coords = tf.logical_and(new_tags == 'N', inactive_tags == 'Y')
    old_coords = tf.where(inactive_tags == 'N', True, False)

    print('new:', new_coords)
    print('inactive:', inactive_coords)
    print('old:', old_coords)

    alpha = 0.25
    gamma = 2.0
    beta = 8.0

    logits = tf.random.normal([8, 1])
    labels = tf.transpose(tf.constant([[1, 0, 1, 0, 1, 1, 1, 0]], dtype=tf.float32))

    predictions = tf.nn.sigmoid(logits)
    zeros = tf.zeros_like(predictions, dtype=predictions.dtype)
    pos = tf.where(labels > zeros, labels - predictions, zeros)
    neg = tf.where(labels > zeros, zeros, predictions * beta)

    fl_loss = -alpha * (pos ** gamma) * tf.math.log(predictions) - (1.0 - alpha) * (neg ** gamma) * tf.math.log(1.0 - predictions)
    fl_loss = tf.squeeze(fl_loss)
    print('fl_loss :', fl_loss)

    new_loss = tf.reduce_mean(fl_loss * tf.cast(new_coords, tf.float32))
    inactive_loss = tf.reduce_mean(fl_loss * tf.cast(inactive_coords, tf.float32))
    old_loss = tf.reduce_mean(fl_loss * tf.cast(old_coords, tf.float32))
    loss = 1.6 * new_loss + 1.2 * inactive_loss + 1.0 * old_loss

    print('new loss: ', new_loss)
    print('inactive loss: ', inactive_loss)
    print('old loss: ', old_loss)
    print('loss :', loss)

    logits = tf.random.normal([8])
    labels = tf.constant([1, 0, 1, 0, 1, 1, 1, 0], dtype=tf.float32)
    loss2 = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
    print(loss2)



def main():
    # test_001()
    # test_002()
    # test_003()
    # kerastune_test()
    # test004()
    # focal_loss_test()
    logical_operator()
    # loss_check()


if __name__ == '__main__':
    main()