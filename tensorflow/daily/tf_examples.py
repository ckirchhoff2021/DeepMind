import numpy as np
import tensorflow as tf

from common_path import *

'''
tensorflow 1.0 daily work
'''


def logistic_test(train=True):
    xn = np.random.randn(100, 2)
    noise = np.random.randn(100, 1) / 1000.0
    yn = np.array([1.0 if v[0] > 0.5 else 0.0 for v in xn])
    print(yn.shape)

    xt = np.array([[0.8, 0.2]])
    yt = np.array([1.0])

    x = tf.placeholder(tf.float64, [None, 2])
    y = tf.placeholder(tf.float64, [None, ])

    w1 = tf.get_variable('w1', [2, 8], dtype=tf.float64, initializer=tf.truncated_normal_initializer(stddev=0.02))
    b1 = tf.get_variable('b1', [8], dtype=tf.float64, initializer=tf.truncated_normal_initializer(stddev=0.02))
    y1 = tf.nn.relu((tf.matmul(x, w1) + b1))

    w2 = tf.get_variable('w2', [8, 1], dtype=tf.float64, initializer=tf.truncated_normal_initializer(stddev=0.02))
    b2 = tf.get_variable('b2', [1], dtype=tf.float64, initializer=tf.truncated_normal_initializer(stddev=0.02))
    y2 = tf.nn.sigmoid(tf.matmul(y1, w2) + b2)

    loss = tf.losses.mean_squared_error(y, tf.squeeze(y2))
    tf.summary.scalar('loss', loss)
    merged_summary = tf.summary.merge_all()

    train_op = tf.train.AdamOptimizer(learning_rate=0.005).minimize(loss)
    init_op = [tf.global_variables_initializer(), tf.local_variables_initializer()]

    xt = np.array([[0.8, 0.2]])
    yt = np.array([1.0])

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init_op)
        if train:
            train_writer = tf.summary.FileWriter(os.path.join(output_path, 'summary'), sess.graph)
            for epoch in range(100):
                _, f_loss, train_summary = sess.run([train_op, loss, merged_summary], feed_dict={x: xn, y: yn})
                train_writer.add_summary(train_summary, epoch)
                print('==> epoch: %d, loss = %f' % (epoch, f_loss))
                saver.save(sess, os.path.join(output_path,'logit/logistic.ckpt'), global_step=epoch)
        else:
            model_file = tf.train.latest_checkpoint(os.path.join(output_path,'logit'))
            saver.restore(sess, model_file)
            pred = sess.run(y2, feed_dict={x: xt, y: yt})
            print('prediction: ', pred)


def test002():

    def input_fn():
        xt = np.random.randn(100, 2)
        yt = np.array([1 if v[0] + v[1] > 0.5 else 0 for v in xt])
        xt = tf.convert_to_tensor(xt, dtype=tf.float32)
        yt = tf.convert_to_tensor(yt, dtype=tf.int32)
        print('==> xt shape: ', xt.shape)
        print('==> yt shape: ', yt.shape)
        return xt, yt

    def model_fn(x, y):
        w1 = tf.get_variable('w1', shape=[2, 10], dtype=tf.float32,
                             initializer=tf.truncated_normal_initializer(stddev=0.02))
        b1 = tf.get_variable('b1', shape=[10], dtype=tf.float32,
                             initializer=tf.truncated_normal_initializer(stddev=0.02))

        w2 = tf.get_variable('w2', shape=[10, 2], dtype=tf.float32,
                             initializer=tf.truncated_normal_initializer(stddev=0.02))
        b2 = tf.get_variable('b2', shape=[2], dtype=tf.float32,
                             initializer=tf.truncated_normal_initializer(stddev=0.02))

        y1 = tf.nn.relu(tf.matmul(x, w1) + b1)
        y2 = tf.matmul(y1, w2) + b2
        loss = tf.losses.sparse_softmax_cross_entropy(y, y2)
        optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)
        return loss, optimizer

    xt, yt = input_fn()
    loss, optimizer = model_fn(xt, yt)
    init_op = [tf.global_variables_initializer(), tf.local_variables_initializer()]
    sess = tf.Session()
    sess.run(init_op)
    for epoch in range(1000):
        vloss, _ = sess.run([loss, optimizer])
        print('==> epoch: %d, loss: %f' % (epoch, vloss))


def batch_test():
    sample_num = 5
    epoch_num = 3
    batch_size = 3
    batch_total = int(sample_num/batch_size) + 1

    def generate_data(sample_num=sample_num):
        labels = np.asarray(range(0, sample_num))
        images = np.random.random([sample_num, 224, 224, 3])
        print('image size {},label size :{}'.format(images.shape, labels.shape))
        return images, labels


    def get_batch_data(batch_size=batch_size):
        images, label = generate_data(sample_num=3)
        images = tf.cast(images, tf.float32)
        labels = tf.cast(label, tf.int32)
        input_queue = tf.train.slice_input_producer([images, labels], shuffle=False, num_epochs=None)
        image_batch, label_batch = tf.train.batch(input_queue, batch_size=batch_size, num_threads=2, capacity=64, allow_smaller_final_batch=False)
        return image_batch, label_batch

    image_batch, label_batch = get_batch_data()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord)
        try:
            for i in range(epoch_num):
                print('********')
                for j in range(batch_total):
                    image_batch_v, label_batch_v = sess.run([image_batch, label_batch])
                    print(image_batch_v.shape, label_batch_v)
        except tf.errors.OutOfRangeError:
            print("done")
        finally:
            coord.request_stop()
        coord.join(threads)


def dataset_test():
    y1 = [1,2,3,4,5,6,7,8,9,10],
    y2 = [11,12,13,14,15,16,17,18,19,20]

    x2 = {
        'A':{
            'data': [1,2,3,4,5,6,7,8,9,10]
        },
        'B': {
            'data': [11,12,13,14,15,16,17,18,19,20]
        },
        'C': {
            'data': [21,22,23,24,25,26,27,28,29,30]
        }
    }

    # dataset = tf.data.Dataset.from_tensor_slices((x1, [0,1,2,3,4,5,6,7,8,9]))
    dataset = tf.data.Dataset.from_tensor_slices((y1,y2))
    dataset = dataset.shuffle(buffer_size=1000).batch(4).repeat(3)
    # dataset = tf.data.Dataset.from_tensor_slices((['aa','bb','cc','dd'],[0,0,1,0]))
    iterator = dataset.make_one_shot_iterator()
    element = iterator.get_next()

    with tf.Session() as sess:
        try:
            while True:
                print(sess.run(element))
        except tf.errors.OutOfRangeError:
            print('end!')

def estimator_test():
    def input_fn1():
        x = np.random.randn(100, 2)
        y = np.asarray([1 if v[0] + v[1] > 0.5 else 0 for v in x])
        dataset = tf.data.Dataset.from_tensor_slices((x, y)).batch(10, drop_remainder=False)
        iterator = dataset.make_one_shot_iterator()
        element = iterator.get_next()
        return element

    def input_fn2():
        x = np.random.randn(50, 2)
        y = np.asarray([1 if v[0] + v[1] > 0.5 else 0 for v in x])
        dataset = tf.data.Dataset.from_tensor_slices((x,y)).batch(2)
        return dataset.make_one_shot_iterator().get_next()

    def model_fn(features, labels, mode, params=None):
        y = tf.layers.dense(features, 10, activation=tf.nn.relu)
        logits = tf.layers.dense(y, 2)
        softmax_logits = tf.nn.softmax(logits)
        predicts = tf.argmax(softmax_logits, axis=1)

        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions = {'predicts': predicts}
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)

        loss = tf.losses.sparse_softmax_cross_entropy(labels, logits)
        accuracy = tf.metrics.accuracy(labels, predicts)
        metrics = {'accuracy': accuracy}

        tf.summary.scalar('train_loss', loss)
        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op, eval_metric_ops=metrics)

    model = tf.estimator.Estimator(model_fn=model_fn,model_dir=os.path.join(output_path, 'estimator'))
    NUM_EPOCHS = 20
    for i in range(NUM_EPOCHS):
        model.train(input_fn1)

    predictions = model.predict(input_fn2)
    for x in predictions:
        print(x)
    print(model.evaluate(input_fn2))



def minist_test():
    mnist = tf.keras.datasets.mnist
    (x1, y1), (x2, y2) = mnist.load_data()
    x1, x2 = x1 / 255.0, x2 /255.0

    def input_fn1():
        dataset = tf.data.Dataset.from_tensor_slices((x1, y1))
        dataset= dataset.shuffle(buffer_size=1000).batch(128,drop_remainder=False)
        iterator = dataset.make_one_shot_iterator()
        return iterator.get_next()

    def input_fn2():
        dataset = tf.data.Dataset.from_tensor_slices((x2, y2))
        dataset = dataset.shuffle(buffer_size=100).batch(64, drop_remainder=False)
        iterator = dataset.make_one_shot_iterator()
        return iterator.get_next()

    def model_fn(features, labels, mode, params=None):
        feature_flatten = tf.layers.flatten(features)
        yt = tf.cast(labels, tf.int32)
        y1 = tf.layers.dense(feature_flatten, 20, activation=tf.nn.relu)
        y2 = tf.layers.dense(y1, 10)
        y_probs = tf.nn.softmax(y2, axis=1)
        y_predicts = tf.argmax(y2, axis=1)

        if mode == tf.estimator.ModeKeys.PREDICT:
            predicts = { 'predicts': y_predicts }
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predicts)

        loss = tf.losses.sparse_softmax_cross_entropy(yt, y2)
        if mode == tf.estimator.ModeKeys.EVAL:
            accuracy = tf.metrics.accuracy(yt, y_predicts)
            metrics = {'accuracy': accuracy}
            return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

        optimizer = tf.train.AdamOptimizer(learning_rate=0.02)
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    model = tf.estimator.Estimator(model_fn, model_dir=os.path.join(output_path, 'mnist'))
    for epoch in range(1):
        model.train(input_fn1)

    print(model.evaluate(input_fn2))


def minist_test_x():
    tf.logging.set_verbosity(tf.logging.INFO)
    mnist = tf.keras.datasets.mnist
    (x1, y1), (x2, y2) = mnist.load_data()
    x1, x2 = x1 / 255.0, x2 /255.0

    def input_fn1():
        dataset = tf.data.Dataset.from_tensor_slices((x1, y1))
        dataset= dataset.shuffle(buffer_size=1000).batch(128,drop_remainder=False)
        iterator = dataset.make_one_shot_iterator()
        return iterator.get_next()

    def input_fn2():
        dataset = tf.data.Dataset.from_tensor_slices((x2, y2))
        dataset = dataset.shuffle(buffer_size=100).batch(64, drop_remainder=False)
        iterator = dataset.make_one_shot_iterator()
        return iterator.get_next()

    def model_fn(features, labels, mode):
        feature_flatten = tf.layers.flatten(features)
        yt = tf.cast(labels, tf.int32)
        y1 = tf.layers.dense(feature_flatten, 20, activation=tf.nn.relu)
        y2 = tf.layers.dense(y1, 10)
        y_probs = tf.nn.softmax(y2, axis=1)
        y_predicts = tf.argmax(y2, axis=1)

        if mode == tf.estimator.ModeKeys.PREDICT:
            predicts = { 'predicts': y_predicts }
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predicts)

        loss = tf.losses.sparse_softmax_cross_entropy(yt, y2)
        if mode == tf.estimator.ModeKeys.EVAL:
            accuracy = tf.metrics.accuracy(yt, y_predicts)
            metrics = {'accuracy': accuracy}
            return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

        optimizer = tf.train.AdamOptimizer(learning_rate=0.02)
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    session_config = tf.ConfigProto(allow_soft_placement=True)
    run_config = tf.estimator.RunConfig(
        model_dir=os.path.join(output_path, 'estimator'),
        session_config=session_config,
        save_checkpoints_steps=2,
        keep_checkpoint_max=10,
        save_summary_steps=1,
        log_step_count_steps=2
    )

    model = tf.estimator.Estimator(model_fn, config=run_config)
    train_input_fn = input_fn1
    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=10)
    eval_input_fn = input_fn2
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn, start_delay_secs=3, throttle_secs=3)

    try:
        tf.estimator.train_and_evaluate(model, train_spec, eval_spec)
    except:
        print('done...')


def mnist_classify():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)
    print(x_train.shape, x_test.shape)
    
    x = tf.placeholder(tf.float32, shape=[None, 784])
    y = tf.placeholder(tf.int64, shape=[None])

    w1 = tf.get_variable('w1', shape=[784, 20], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
    b1 = tf.get_variable('b1', shape=[20], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
    w2 = tf.get_variable('w2', shape=[20, 10], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
    b2 = tf.get_variable('b2', shape=[10], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))

    y1 = tf.nn.relu(tf.matmul(x, w1) + b1)
    y2 = tf.matmul(y1, w2) + b2
    yp = tf.argmax(y2,axis=1)
    acc = tf.reduce_mean(tf.cast(tf.equal(yp, y), tf.float32))
    loss = tf.losses.sparse_softmax_cross_entropy(y, y2)

    tf.summary.scalar('loss', loss)
    tf.summary.scalar('acc', acc)
    merged_summary = tf.summary.merge_all()
    saver = tf.train.Saver()
    optimizer = tf.train.AdamOptimizer(learning_rate=0.005).minimize(loss)
    init_op = [tf.global_variables_initializer(), tf.local_variables_initializer()]
    sess = tf.Session()
    sess.run(init_op)

    best_acc = 0.0
    for epoch in range(5):
        train_writer = tf.summary.FileWriter(os.path.join(output_path, 'summary'), sess.graph)
        batch_size = 128
        batches = int(x_train.shape[0] / batch_size) + 1
        train_loss = 0.0
        train_acc = 0.0
        for i in range(batches):
            k1 = batch_size * i
            k2 = batch_size * (i+1)
            k2 = k2 if k2 < x_train.shape[0] else x_train.shape[0]
            xf, yf = x_train[k1:k2], y_train[k1:k2]
            _, batch_loss, batch_acc, summary = sess.run([optimizer, loss, acc, merged_summary], feed_dict={x:xf,y:yf})
            train_writer.add_summary(summary, epoch * batches + i)
            if i % 100 == 0:
                print('==> epoch: [%d]/[%d] - [%d]/[%d], training loss = %f, acc = %f' % (epoch, 5, i, batches, batch_loss, batch_acc))
            train_loss += batch_loss
            train_acc += batch_acc

        train_loss = train_loss / batches
        train_acc = train_acc / batches
        print('************************************************************')
        print('* epoch: %d, training avaerage_loss = %f, average_acc = %f' %(epoch, train_loss, train_acc))
        # print('*************************************************************')

        batch_size = 64
        batches = int(x_test.shape[0] / batch_size) + 1
        test_loss = 0.0
        test_acc = 0.0
        for i in range(batches):
            k1 = batch_size * i
            k2 = batch_size * (i + 1)
            k2 = k2 if k2 < x_test.shape[0] else x_test.shape[0]
            xf, yf = x_test[k1:k2], y_test[k1:k2]
            batch_loss, batch_acc = sess.run([loss, acc], feed_dict={x: xf, y: yf})
            test_loss += batch_loss
            test_acc += batch_acc
        test_loss = test_loss / batches
        test_acc = test_acc / batches
        # print('************************************************************')
        print('* epoch: %d, testing avaerage_loss = %f, average_acc = %f' % (epoch, test_loss, test_acc))
        print('*************************************************************')
        
        if best_acc < test_acc:
            best_acc = test_acc
            saver.save(sess, os.path.join(output_path, 'mnist_cls.ckpt'), global_step=epoch)


def mnist_classify2():
    mnist = tf.keras.datasets.mnist
    (x1, y1), (x2, y2) = mnist.load_data()
    x1 = x1.reshape(x1.shape[0], -1) / 255.0
    x2 = x2.reshape(x2.shape[0], -1) / 255.0

    def _make_dataset_(x, y, batch_size=128, repeat=1):
        datas = tf.data.Dataset.from_tensor_slices((x, y))
        datas = datas.shuffle(buffer_size=1000).batch(batch_size, drop_remainder=False).repeat(5)
        iterator = datas.make_one_shot_iterator()
        element = iterator.get_next()
        return element

    e1 = _make_dataset_(x1, y1, batch_size=128, repeat=2)
    e2 = _make_dataset_(x2, y2, batch_size=64, repeat=-1)

    def model_fn(x, y):
        yt = tf.cast(y, tf.int64)
        y1 = tf.layers.dense(x, 20, activation=tf.nn.relu)
        y2 = tf.layers.dense(y1, 10)
        yp = tf.argmax(y2, axis=1)
        acc = tf.reduce_mean(tf.cast(tf.equal(yp, yt), tf.float32))
        loss = tf.losses.sparse_softmax_cross_entropy(yt, y2)
        return loss, acc

    x = tf.placeholder(tf.float32, [None, 784])
    y = tf.placeholder(tf.int32, [None])

    loss, acc = model_fn(x, y)
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('acc', acc)
    merged_summary = tf.summary.merge_all()

    saver = tf.train.Saver()
    optimizer = tf.train.AdamOptimizer(learning_rate=0.005).minimize(loss)
    init_op = [tf.global_variables_initializer(), tf.local_variables_initializer()]
    sess = tf.Session()
    sess.run(init_op)
    train_writer = tf.summary.FileWriter(os.path.join(output_path, 'summary'), sess.graph)

    best_acc = 0.0
    icount = 0
    try:
        while True:
            icount += 1
            batch_x, batch_y = sess.run(e1)
            _, batch_loss, batch_acc, summary = sess.run([optimizer, loss, acc, merged_summary], feed_dict={x:batch_x, y:batch_y})
            train_writer.add_summary(summary, icount)
            if icount % 100 == 0:
                print('==> ibatch: [%d], training loss = %f, acc = %f' % (icount, batch_loss, batch_acc))

            if icount % 200 == 0:
                test_loss = 0.0
                test_acc = 0.0
                for i in range(100):
                    batch_x, batch_y = sess.run(e2)
                    batch_loss, batch_acc = sess.run([loss, acc], feed_dict={x: batch_x, y: batch_y})
                    test_loss += batch_loss
                    test_acc += batch_acc
                test_loss = test_loss / 100.0
                test_acc = test_acc / 100.0
                print('==> ibatch: [%d]: testing avaerage_loss = %f, average_acc = %f' % (icount, test_loss, test_acc))
                if best_acc < test_acc:
                    best_acc = test_acc
                    saver.save(sess, os.path.join(output_path, 'mnist_cls.ckpt'), global_step=icount)


    except tf.errors.OutOfRangeError:
        print('end!')




'''
tf2.0
import tensorflow as tf
from tensorflow.python.keras.models import Model


@tf.function
def inner_fucntion(x,y,b):
    x = tf.matmul(x,y)
    x = x + b
    return x


def main():
    x1 = tf.constant([1.0,2.0],dtype=tf.float32)
    x2 = tf.constant([1.1,2.2], dtype=tf.float32)
    x3 = x1 + x2
    print(x3)

    x4 = tf.reshape(x1, [1,2])
    x5 = tf.reshape(x2, [2, 1])
    print(x4 * x5)

    y1 = tf.ones((2,2))
    with tf.GradientTape() as t:
        t.watch(y1)
        y2 = tf.reduce_sum(y1)
        z = y2 * y2

    dz_dy1 = t.gradient(z, y1)
    print(dz_dy1)
    print(y2)

    g_function = tf.function(inner_fucntion)
    x1 = tf.constant([[1.0, 2.0]])
    y1 = tf.constant([[2.0], [3.0]])
    b1 = tf.constant(4.0)
    print(g_function(x1, y1, b1))


class Dense(tf.Module):
    def __init__(self, in_features, out_features):
        super(Dense, self).__init__()
        self.w = tf.Variable(tf.random.normal([in_features, out_features], name='w'))
        self.b = tf.Variable(tf.zeros([out_features]),name='b')
    
    def __call__(self, x):
        y = tf.matmul(x, self.w) + self.b
        return y


class LinearModel(tf.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.fc1 = Dense(in_features=3, out_features=3)
        self.fc2 = Dense(in_features=3, out_features=2)

    def __call__(self, x):
        y1 = self.fc1(x)
        y2 = self.fc2(y1)
        return y2


if __name__ == '__main__':
    net = LinearModel()
    x1 = tf.random.uniform([2,3])
    print(x1)
    y1 = net(x1)
    print(y1)

'''

def placeholder_test():
    x = tf.placeholder(tf.float32, shape=[None, 2])
    x1 = [[1.0, 2.0]]
    x2 = [[3.0, 4.0]]
    y1 = x + 0.1
    y2 = x + 0.2
    k = y2 - y1
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    v1 = sess.run([y1], feed_dict={x:x1})
    v2 = sess.run([y2], feed_dict={x:x2})
    print(v1, v2)


def sample_test():
    candidates = np.array([[1,1,2,2,2,3,3,3,3,4,4,4,4,4,4,5,5,6,6,7,8,9]])
    print(candidates.shape)
    candidates = np.reshape(candidates, (22, 1))
    print(candidates.shape)
    sampled_ids, true_expected, sample_expected = tf.nn.log_uniform_candidate_sampler(candidates,
                                                                                      num_true=1,
                                                                                      num_sampled=20,
                                                                                      unique=True,
                                                                                      range_max=1000)
    init_op = [tf.global_variables_initializer(), tf.local_variables_initializer()]
    with tf.Session() as sess:
        sess.run(init_op)
        print(sess.run(sampled_ids))
        print(sess.run(true_expected))
        print(sess.run(sample_expected))




if __name__ == '__main__':
    # logistic_test(train=False)
    # test002()
    # batch_test()
    # dataset_test()
    # estimator_test()
    # minist_test()
    # mnist_classify()
    # mnist_classify2()
    # placeholder_test()
    # sample_test()
    minist_test_x()