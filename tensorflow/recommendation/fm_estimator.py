import numpy as np
import tensorflow as tf
from common_path import *
from fm_regression import RatingDataset
from layers import *
from estimator_model import *


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

    run_config = tf.estimator.RunConfig(model_dir=os.path.join(output_path, 'regressor'), save_checkpoints_steps=100, log_step_count_steps=2)
    model = EmbeddingNet(feature_columns, run_config, hidden_units)

    def input_fn():
        dataset = tf.data.Dataset.from_tensor_slices((sample_dict, labels))
        dataset = dataset.shuffle(1000).repeat(1).batch(128)
        return dataset

    model.train(input_fn=input_fn)


def cls_test():
    tf.logging.set_verbosity(tf.logging.INFO)
    feature_columns = [
        {'feature_name': 'age', 'embedding_dim': 10, 'hash_buckets': 100, 'type': 'int', 'sequence': 1},
        {'feature_name': 'gender', 'embedding_dim': 10, 'hash_buckets': 10, 'type': 'string', 'sequence': 1},
        {'feature_name': 'occupation', 'embedding_dim': 10, 'hash_buckets': 1000, 'type': 'int', 'sequence': 1},
        {'feature_name': 'code', 'embedding_dim': 10, 'hash_buckets': 10000, 'type': 'string', 'sequence': 1},
        {'feature_name': 'genres', 'embedding_dim': 10, 'hash_buckets': 1000, 'type': 'string', 'sequence': 3},
    ]
    hidden_units = [32, 64]
    layer_fn = build_layers(feature_columns, hidden_units)
    datas = RatingDataset()
    train_samples, train_labels, test_samples, test_labels = datas.get_cls_samples()

    run_config = tf.estimator.RunConfig(model_dir=os.path.join(output_path, 'cls'), save_checkpoints_steps=100, log_step_count_steps=10)
    model = EmbeddingNet(feature_columns, run_config, hidden_units)

    def train_input_fn():
        dataset = tf.data.Dataset.from_tensor_slices((train_samples, train_labels))
        dataset = dataset.shuffle(1000).repeat(1).batch(128)
        return dataset

    def eval_input_fn():
        dataset = tf.data.Dataset.from_tensor_slices((test_samples, test_labels))
        dataset = dataset.shuffle(1000).repeat(1).batch(64)
        return dataset

    train_spec = tf.estimator.TrainSpec(train_input_fn, max_steps=10000)
    eval_spec = tf.estimator.EvalSpec(eval_input_fn,start_delay_secs=10, throttle_secs=20)

    tf.estimator.train_and_evaluate(model, train_spec, eval_spec)



def test2():
    tf.logging.set_verbosity(tf.logging.INFO)
    feature_columns = [
        {'feature_name': 'age', 'embedding_dim': 10, 'hash_buckets': 100, 'type': 'int', 'sequence': 1},
        {'feature_name': 'gender', 'embedding_dim': 10, 'hash_buckets': 10, 'type': 'string', 'sequence': 1},
        {'feature_name': 'occupation', 'embedding_dim': 10, 'hash_buckets': 1000, 'type': 'int', 'sequence': 1},
        {'feature_name': 'code', 'embedding_dim': 10, 'hash_buckets': 10000, 'type': 'string', 'sequence': 1},
        {'feature_name': 'genres', 'embedding_dim': 10, 'hash_buckets': 1000, 'type': 'string', 'sequence': 3},
    ]
    hidden_units = [32, 64]
    layer_fn = build_layers(feature_columns, hidden_units)

    eval_datas = {
        'age': np.array([1,2]),
        'gender':np.array(['M','F']),
        'occupation': np.array([112, 256]),
        'code': np.array(['xyz', 'wbs']),
        'genres':np.array([['11','22','33'], ['21', '03', '45']])
    }
    eval_labels = np.array([0.8, 0.6],dtype=np.float32)
    eval_inputs = tf.estimator.inputs.numpy_input_fn(eval_datas, eval_labels, shuffle=False)

    train_datas = {
        'age': np.array([1,1,2,2,3,3,2,3]),
        'gender':np.array(['M','F','M','F','M','F','M','F']),
        'occupation': np.array([112,12,13,22,33,44,12,67]),
        'code': np.array(['xyz','abc','v112','xyz','abc','v112','xyz','abc']),
        'genres':np.array([['11','22','33'],['11','42','43'],['22','12','43'],['44','32','23'],
                           ['44','32','23'],['44','32','23'],['44','32','23'],['44','32','23']])
    }
    train_labels = np.array([1,2,1,3,2,1,1,2], dtype=np.float32)
    train_inputs = tf.estimator.inputs.numpy_input_fn(train_datas, train_labels, shuffle=False, batch_size=2, num_epochs=2)

    run_config = tf.estimator.RunConfig(save_checkpoints_steps=100, model_dir=os.path.join(output_path, 'estimator'))
    model = EmbeddingNet(feature_columns, run_config, hidden_units)
    # model.train(train_inputs)

    pred_inputs = tf.estimator.inputs.numpy_input_fn(eval_datas, shuffle=False)
    values = model.predict(pred_inputs)
    for value in values:
        print(value)


    '''
    feature_op, label_op = train_inputs()
    with tf.Session() as sess:
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        coordinator = tf.train.Coordinator()
        _ = tf.train.start_queue_runners(coord=coordinator)
        try:
            while 1:
                print(sess.run([feature_op, label_op]))
        except tf.errors.OutOfRangeError:
            print('End.')
    '''



def main():
    # regression_test()
    # test2()
    cls_test()

if __name__ == '__main__':
    main()
