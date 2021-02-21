import os
import numpy as np
import tensorflow as tf
from common_path import *

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)


class ClsNet:
    def __init__(self, ndims, ncls=10):
        self.x = tf.placeholder(tf.float32, shape=[None, ndims])
        self.y = tf.placeholder(tf.int64, shape=[None])
        self.w1 = tf.get_variable('w1', shape=[ndims, 20], dtype=tf.float32,
                             initializer=tf.truncated_normal_initializer(stddev=0.02))
        self.b1 = tf.get_variable('b1', shape=[20], dtype=tf.float32,
                             initializer=tf.truncated_normal_initializer(stddev=0.02))
        self.w2 = tf.get_variable('w2', shape=[20, ncls], dtype=tf.float32,
                             initializer=tf.truncated_normal_initializer(stddev=0.02))
        self.b2 = tf.get_variable('b2', shape=[ncls], dtype=tf.float32,
                             initializer=tf.truncated_normal_initializer(stddev=0.02))

        self.saver = tf.train.Saver()
        self.sess = tf.Session()

    def inference(self):
        y1 = tf.nn.relu(tf.matmul(self.x, self.w1) + self.b1)
        y2 = tf.matmul(y1, self.w2) + self.b2
        return y2

    def restore(self, model_dir):
        model_file = tf.train.latest_checkpoint(model_dir)
        self.saver.restore(self.sess, model_file)

    def predict(self, x_input):
        y2 = self.inference()
        yp = tf.argmax(y2, axis=1)
        ret = self.sess.run(yp,feed_dict={self.x:x_input})
        return ret

    def start_train(self, x_train, y_train, x_test, y_test, model_dir, summary_dir, epochs=5):
        y2 = self.inference()
        yp = tf.argmax(y2, axis=1)
        acc = tf.reduce_mean(tf.cast(tf.equal(yp, self.y), tf.float32))
        loss = tf.losses.sparse_softmax_cross_entropy(self.y, y2)

        tf.summary.scalar('loss', loss)
        tf.summary.scalar('acc', acc)
        merged_summary = tf.summary.merge_all()

        optimizer = tf.train.AdamOptimizer(learning_rate=0.005).minimize(loss)
        init_op = [tf.global_variables_initializer(), tf.local_variables_initializer()]
        self.sess.run(init_op)

        best_acc = 0.0
        train_writer = tf.summary.FileWriter(summary_dir, self.sess.graph)
        for epoch in range(epochs):
            batch_size = 128
            batches = int(x_train.shape[0] / batch_size) + 1
            train_loss = 0.0
            train_acc = 0.0
            for i in range(batches):
                k1 = batch_size * i
                k2 = batch_size * (i + 1)
                k2 = k2 if k2 < x_train.shape[0] else x_train.shape[0]
                xf, yf = x_train[k1:k2], y_train[k1:k2]
                _, batch_loss, batch_acc, summary = self.sess.run(
                    [optimizer, loss, acc, merged_summary], feed_dict={self.x: xf, self.y: yf})
                train_writer.add_summary(summary, epoch * batches + i)
                if i % 100 == 0:
                    print('==> epoch: [%d]/[%d] - [%d]/[%d], training loss = %f, acc = %f' % (
                        epoch, epochs, i, batches, batch_loss, batch_acc))
                train_loss += batch_loss
                train_acc += batch_acc

            train_loss = train_loss / batches
            train_acc = train_acc / batches
            print('************************************************************')
            print('* epoch: %d, training avaerage_loss = %f, average_acc = %f' % (epoch, train_loss, train_acc))

            batch_size = 64
            batches = int(x_test.shape[0] / batch_size) + 1
            test_loss = 0.0
            test_acc = 0.0
            for i in range(batches):
                k1 = batch_size * i
                k2 = batch_size * (i + 1)
                k2 = k2 if k2 < x_test.shape[0] else x_test.shape[0]
                xf, yf = x_test[k1:k2], y_test[k1:k2]
                batch_loss, batch_acc = self.sess.run([loss, acc], feed_dict={self.x: xf, self.y: yf})
                test_loss += batch_loss
                test_acc += batch_acc
            test_loss = test_loss / batches
            test_acc = test_acc / batches
            print('* epoch: %d, testing avaerage_loss = %f, average_acc = %f' % (epoch, test_loss, test_acc))
            print('*************************************************************')

            if best_acc < test_acc:
                best_acc = test_acc
                self.saver.save(self.sess, os.path.join(model_dir, 'model.ckpt'), global_step=epoch)


def mnist_classify():
    net = ClsNet(784, 10)
    model_dir = os.path.join(output_path, 'mnist')
    summary_dir = os.path.join(output_path, 'summary')
    # net.start_train(x_train, y_train, x_test, y_test, model_dir, summary_dir)
    net.restore(model_dir)
    yp = net.predict(x_test[12:20])
    print('predict: ', yp)
    print('truth: ', y_test[12:20])


class EmbeddingNet:
    def __init__(self, ndims, nembs):
        self.x1 = tf.placeholder(tf.float32, shape=[None, ndims])
        self.x2 = tf.placeholder(tf.float32, shape=[None, ndims])
        self.x3 = tf.placeholder(tf.float32, shape=[None, ndims])
        self.variable_dict = dict()

        weights_names = ['w1','b1','w2','b2']
        weights_shapes = [(ndims, 64), (64), (64, nembs), (nembs)]
        for i in range(len(weights_names)):
            name = weights_names[i]
            shape = weights_shapes[i]
            variable = tf.get_variable(name, shape=shape, dtype=tf.float32,
                                       initializer=tf.truncated_normal_initializer())

            self.variable_dict[name] = variable

        self.saver = tf.train.Saver()
        self.sess = tf.Session()

    def inference(self, x):
        w1 = self.variable_dict['w1']
        b1 = self.variable_dict['b1']
        w2 = self.variable_dict['w2']
        b2 = self.variable_dict['b2']
        y1 = tf.nn.relu(tf.matmul(x, w1) + b1)
        y2 = tf.nn.tanh(tf.matmul(y1, w2) + b2)
        vec = tf.nn.l2_normalize(y2)
        return vec

    def restore(self, model_dir):
        model_file = tf.train.latest_checkpoint(model_dir)
        self.saver.restore(self.sess, model_file)

    def predict(self, x_input):
        vec = self.inference(self.x1)
        ret = self.sess.run(vec, feed_dict={self.x1: x_input})
        return ret

    def triplet_loss(self, margin=0.05):
        e1 = self.inference(self.x1)
        e2 = self.inference(self.x2)
        e3 = self.inference(self.x3)

        d1 = tf.reduce_sum(e1 * e2, 1)
        d2 = tf.reduce_sum(e1 * e3, 1)
        loss = tf.maximum(0.0, margin + d1 - d2)
        loss = tf.reduce_mean(loss)
        return loss

    def start_train(self, anchors, positives, negatives, model_dir, summary_dir, epochs=5):
        print('-- start training loop --')

        loss = self.triplet_loss()
        tf.summary.scalar('loss', loss)
        merged_summary = tf.summary.merge_all()

        optimizer = tf.train.AdamOptimizer(learning_rate=0.05).minimize(loss)
        init_op = [tf.global_variables_initializer(), tf.local_variables_initializer()]
        self.sess.run(init_op)

        train_writer = tf.summary.FileWriter(summary_dir, self.sess.graph)
        for epoch in range(epochs):
            batch_size = 128
            batches = int(anchors.shape[0] / batch_size) + 1
            train_loss = 0.0
            for i in range(batches):
                k1 = batch_size * i
                k2 = batch_size * (i + 1)
                k2 = k2 if k2 < anchors.shape[0] else anchors.shape[0]
                x1f, x2f, x3f = anchors[k1:k2], positives[k1:k2], negatives[k1:k2]
                _, batch_loss, summary = self.sess.run(
                    [optimizer, loss, merged_summary], feed_dict={self.x1: x1f, self.x2: x2f, self.x3:x3f})
                train_writer.add_summary(summary, epoch * batches + i)
                if i % 100 == 0:
                    print('==> epoch: [%d]/[%d] - [%d]/[%d], training loss = %f' % (
                        epoch, epochs, i, batches, batch_loss))
                train_loss += batch_loss

            train_loss = train_loss / batches
            print('****** epoch: %d, training avaerage_loss = %f' % (epoch, train_loss))
            self.saver.save(self.sess, os.path.join(model_dir, 'model.ckpt'), global_step=epoch)


def mnist_embedding():
    net = EmbeddingNet(784, 32)
    anchors = list()
    positives = list()
    negatives = list()

    index_dict = dict()
    for index, y in enumerate(y_train):
        if y not in index_dict.keys():
            index_dict[y] = list()
        index_dict[y].append(index)

    for index, y in enumerate(y_train):
        '''
        if index > 10:
            break
        '''
        pos_list = index_dict[y]
        anchor = index
        pos = np.random.choice(pos_list, 3)
        neg_ys = list(set(range(10)) - set([y]))
        neg_y = np.random.choice(neg_ys, 1)[0]
        neg_list = index_dict[neg_y]
        neg = np.random.choice(neg_list, 3)

        anchors.append(x_train[index])
        positives.append(x_train[pos[1]])
        negatives.append(x_train[neg[1]])
    
    anchors = np.asarray(anchors)
    positives = np.asarray(positives)
    negatives = np.asarray(negatives)

    print('anchor: ', anchors.shape)
    print('positive: ', positives.shape)
    print('negative: ', negatives.shape)

    model_dir = os.path.join(output_path, 'mnist')
    summary_dir = os.path.join(output_path, 'summary')
    # net.start_train(anchors, positives, negatives, model_dir, summary_dir)

    net.restore(model_dir)
    yp = net.predict(x_test[12:13])
    print('embedding: ', yp)



if __name__ == '__main__':
    # mnist_classify()
    mnist_embedding()
