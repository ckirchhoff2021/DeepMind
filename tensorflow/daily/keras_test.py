import tensorflow as tf
from tensorflow import keras


def keras_cls():
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    print(train_images.shape)



def main():
    print(tf.__version__)
    keras_cls()


if __name__ == '__main__':
    main()