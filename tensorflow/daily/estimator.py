import numpy as np
import pandas as pd
import tensorflow as tf
from common_path import *
from sklearn.datasets import load_iris


def estimator_cls():
    tf.logging.set_verbosity(tf.logging.INFO)
    data = load_iris()
    x = data.data
    y = data.target

    feature_columns = []
    '''
    train_inputs = tf.estimator.inputs.numpy_input_fn(
        {'x1': x[:,0], 'x2': x[:,1], 'x3': x[:,2], 'x4':x[:,3]},y=y, batch_size=8,  shuffle=True, num_epochs=10)
    for v in ['x1', 'x2', 'x3', 'x4']:
        feature_columns.append(tf.feature_column.numeric_column(key=v))
    '''

    train_inputs = tf.estimator.inputs.numpy_input_fn(
        {'x': x}, y=y, batch_size=8, shuffle=True, num_epochs=10)

    feature_columns.append(tf.feature_column.numeric_column(key='x', shape=(4)))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    model = tf.estimator.DNNClassifier([8,4], feature_columns, n_classes=3, optimizer=optimizer,
                                       model_dir=os.path.join(output_path, 'estimator'))

    # model.train(train_inputs)

    predict_inputs = tf.estimator.inputs.numpy_input_fn({'x': np.array([[0.4,0.8,0.8,0.2]])}, shuffle=False)
    values = model.predict(predict_inputs)
    for v in values:
        print(v)


def create_model(features, feature_columns, hidden_units, output_cls):
    inputs = tf.feature_column.input_layer(features=features, feature_columns=feature_columns)
    for units in hidden_units:
        inputs = tf.layers.dense(inputs=inputs, units=units, activation=tf.nn.relu)
    logits = tf.layers.dense(inputs=inputs, units=output_cls)
    return logits


def model_fn_builder(lr):
    def model_fn(features, labels, mode, params, config):
        logits = create_model(features, params['feature_columns'], params['hidden_units'], params['output_cls'])
        predicts = tf.argmax(input=logits, axis=1)
        probs = tf.nn.softmax(logits=logits, axis=1)

        is_predict = (mode == tf.estimator.ModeKeys.PREDICT)

        if not is_predict:
            loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
            def metric_fn(labels, predictions):
                accuracy, accuracy_update = tf.metrics.accuracy(labels=labels, predictions=predictions, name='iris_accuracy')
                recall, recall_update = tf.metrics.recall(labels=labels, predictions=predictions, name='iris_recall')
                precision, precision_update = tf.metrics.precision(labels=labels, predictions=predictions, name='iris_precision')
                return {
                    'accuracy': (accuracy, accuracy_update),
                    'recall': (recall, recall_update),
                    'precision': (precision, precision_update)
                }

            if mode == tf.estimator.ModeKeys.EVAL:
                return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=metric_fn(labels, predicts))

            train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss=loss,global_step=tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op, eval_metric_ops=metric_fn(labels, predicts))
        else:
            predictions = {'predicts': predicts, 'probabilities': probs}
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    return model_fn


def input_fn_builder(file_path, batch_size, epochs):
    def parse_line(line):
        csv_types = [[0.0], [0.0], [0.0], [0.0], [0]]
        csv_column_names = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'label']
        fields = tf.decode_csv(line, record_defaults=csv_types)
        features = dict(zip(csv_column_names, fields))
        labels = features.pop('label')
        return features, labels
    
    def input_fn():
        dataset = tf.data.TextLineDataset(file_path).skip(1)
        dataset = dataset.map(parse_line)
        dataset = dataset.shuffle(1000).repeat(epochs).batch(batch_size)
        return dataset

    return input_fn


def iris_test():
    tf.logging.set_verbosity(tf.logging.INFO)
    csv_types = [[0.0], [0.0], [0.0], [0.0], [0]]
    csv_column_names = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'label']
    labels = ['Setosa', 'Versicolor', 'Virginica']
    batch_size = 16
    epochs = 200

    feature_columns = []
    for i in range(len(csv_column_names)-1):
        feature_columns.append(tf.feature_column.numeric_column(key=csv_column_names[i]))
    ncls = len(labels)
    hidden_units = [128,256,256]
    params = dict()
    params['feature_columns'] = feature_columns
    params['hidden_units'] = hidden_units
    params['output_cls'] = ncls

    config = tf.estimator.RunConfig(save_checkpoints_steps=100)
    model = tf.estimator.Estimator(model_fn=model_fn_builder(0.001), model_dir=os.path.join(output_path, 'iris'),
                                   params=params, config=config)
    # model.train(input_fn=input_fn_builder(os.path.join(data_path, 'iris_training.csv'), batch_size, epochs))
    model.evaluate(input_fn=input_fn_builder(os.path.join(data_path, 'iris_test.csv'), 2, 1))


def embedding_test():
    feature_dict = {
        'fx': ['ac', 'exr'],
        'fy': ['axe', 'wst']
    }
    labels = [0, 1]
    feature_names = ['fx', 'fy']
    variable_dict = dict()
    for name in feature_names:
        variable = tf.get_variable(name, shape=[20, 10], dtype=tf.float32,
                                    initializer=tf.truncated_normal_initializer(stddev=0.02))
        variable_dict[name] = variable

    def process(feature, label):
        feature_dict = dict()
        for name in feature_names:
            indices = tf.strings.to_hash_bucket_fast(feature[name], 20)
            embeddings = tf.nn.embedding_lookup(variable_dict[name], indices)
            feature_dict[name] = embeddings
        return feature_dict, label

    dataset = tf.data.Dataset.from_tensor_slices((feature_dict, labels))
    dataset = dataset.map(process)
    # iterator = dataset.make_one_shot_iterator()
    iterator = dataset.make_initializable_iterator()
    element = iterator.get_next()

    sess = tf.Session()
    init_op = [tf.global_variables_initializer(), tf.local_variables_initializer()]
    sess.run(init_op)
    sess.run(iterator.initializer)
    try:
        while 1:
            print(sess.run(element))
    except tf.errors.OutOfRangeError:
        print('End!')



def main():
    # estimator_cls()
    # iris_test()
    embedding_test()


if __name__ == '__main__':
    main()