import tensorflow as tf
from common_path import *
from fm_regression import RatingDataset


class RegressorNet:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.variable_dict = dict()
        self.initialize()

    def initialize(self):
        for name in self.config.keys():
            params = self.config[name]
            variable = tf.get_variable(name, shape=[params['buckets'], params['dim']], dtype=tf.float32,
                                       initializer=tf.truncated_normal_initializer(stddev=0.002))
            self.variable_dict[name] = variable

    def get_embedding(self, feature_dict):
        embedding_list = list()
        for colname in self.config.keys():
            params = self.config[colname]
            if params['type'] == 'string':
                indices = tf.strings.to_hash_bucket_fast(feature_dict[colname], params['buckets'])
                embedding = tf.nn.embedding_lookup(self.variable_dict[colname], indices)
            else:
                embedding = tf.nn.embedding_lookup(feature_dict[colname], sample_dict[colname])
            embedding = tf.layers.flatten(embedding)
            embedding_list.append(embedding)
        embedding = tf.concat(embedding_list, 1)
        return embedding

    def process(self, feature_dict, labels):
        embedding = self.get_embedding(feature_dict)
        return embedding, labels


def model_fn(features, labels, mode, config):
    y1 = tf.layers.dense(features, 128, activation=tf.nn.relu)
    y2 = tf.layers.dense(y1, 1, activation=tf.nn.sigmoid)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predicts = {'predicts': y2}
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predicts)

    yt = tf.cast(labels, tf.float32)
    loss = tf.losses.mean_squared_error(yt, y2)
    mse = tf.metrics.mean_squared_error(yt, y2)
    metrics = {'mse': mse}

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

    optimizer = tf.train.AdamOptimizer(learning_rate=0.005)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op, eval_metric_ops=metrics)


def regression_test():
    config = {
        'age': {'dim': 10, 'buckets': 100, 'type': 'int', 'sequence': 1},
        'gender': {'dim': 10, 'buckets': 10, 'type': 'string', 'sequence': 1},
        'occupation': {'dim': 10, 'buckets': 1000, 'type': 'int', 'sequence': 1},
        'code': {'dim': 10, 'buckets': 10000, 'type': 'string', 'sequence': 1},
        'genres': {'dim': 10, 'buckets': 1000, 'type': 'string', 'sequence': 3},
    }

    datas = RatingDataset()
    sample_dict, labels = datas.get_samples()

    def input_fn():
        dataset = tf.data.Dataset.from_tensor_slices((sample_dict, labels))
        dataset.map(net.process)
        dataset = dataset.shuffle(buffer_size=1000).batch(128, drop_remainder=False)
        iterator = dataset.make_one_shot_iterator()
        return iterator.get_next()

    net = RegressorNet(config)
    model = tf.estimator.Estimator(model_fn, model_dir=os.path.join(output_path, 'estimator'))
    model.train(input_fn)



def main():
    regression_test()


if __name__ == '__main__':
    main()