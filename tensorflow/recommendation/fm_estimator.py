import tensorflow as tf
from common_path import *
from fm_regression import RatingDataset
from layers import *


def create_model(features, feature_columns, hidden_units, output_cls):
    inputs = tf.feature_column.input_layer(features=features, feature_columns=feature_columns)
    for unit in hidden_units:
        inputs = tf.layers.dense(inputs, unit, activation=tf.nn.relu)
    logits = tf.layers.dense(inputs, output_cls, activation=tf.nn.sigmoid)
    return logits

def init_variables(config):
    variable_dict = dict()
    for name in config.keys():
        params = config[name]
        variable = tf.get_variable(name, shape=[params['buckets'], params['dim']], dtype=tf.float32,
                                   initializer=tf.truncated_normal_initializer(stddev=0.002))
        variable_dict[name] = variable
    return variable_dict


def model_fn_builder(lr, layer_fn):
    def model_fn(features, labels, mode, config):
        embedding = layer_fn(features)
        logits = tf.layers.dense(embedding, 1, activation=tf.nn.sigmoid)

        is_predict = (mode == tf.estimator.ModeKeys.PREDICT)
        if not is_predict:
            loss = tf.losses.mean_squared_error(labels=labels, logits=logits)
            mse = tf.metrics.mean_squared_error(labels=labels, predictions=logits)
            metrics = {'mse': mse}
            if mode == tf.estimator.ModeKeys.EVAL:
                return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=metric_fn(labels, logits))

            train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss=loss, global_step=tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op, eval_metric_ops=metrics)
        else:
            predictions = {'predicts':logits}
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    return model_fn


def regression_test():
    tf.logging.set_verbosity(tf.logging.INFO)
    feature_columns = [
        {'feature_name':'age', 'embedding_dim': 10, 'hash_buckets': 100, 'type': 'int', 'sequence': 1},
        {'feature_name':'gender','embedding_dim': 10, 'hash_buckets': 10, 'type': 'string', 'sequence': 1},
        {'feature_name':'occupation','embedding_dim': 10, 'hash_buckets': 1000, 'type': 'int', 'sequence': 1},
        {'feature_name':'code', 'embedding_dim': 10, 'hash_buckets': 10000, 'type': 'string', 'sequence': 1},
        {'feature_name':'genres', 'embedding_dim': 10, 'hash_buckets': 1000, 'type': 'string', 'sequence': 3},
    ]
    hidden_units = [32, 64]
    layer_fn = build_layers(feature_columns, hidden_units)
    datas = RatingDataset()
    sample_dict, labels = datas.get_samples()
    model_fn = model_fn_builder(0.001, layer_fn)

    config = tf.estimator.RunConfig(save_checkpoints_steps=100)
    model = tf.estimator.Estimator(model_fn=model_fn, model_dir=os.path.join(output_path, 'regressor'), config=config)

    def input_fn():
        dataset = tf.data.Dataset.from_tensor_slices((sample_dict, labels))
        dataset = dataset.shuffle(1000).repeat(1).batch(128)
        # iterator = dataset.make_one_shot_iterator()
        # return iterator.get_next()
        return dataset

    model.train(input_fn=input_fn)



def main():
    regression_test()


if __name__ == '__main__':
    main()