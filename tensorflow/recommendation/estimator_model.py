import tensorflow as tf

class EmbeddingNet(tf.estimator.Estimator):
    def __init__(self, feature_config, run_config, hidden_units, lr=0.005):
        self.feature_columns = feature_config
        self.hidden_units = hidden_units

        def _model_fn_(features, labels, mode, config):
            embedding = self.layer_fn(features)
            logits = tf.layers.dense(embedding, 1, activation=tf.nn.sigmoid)
            logits = tf.squeeze(logits)
            is_predict = (mode == tf.estimator.ModeKeys.PREDICT)
            if not is_predict:
                loss = tf.losses.mean_squared_error(labels=labels, predictions=logits)
                mse = tf.metrics.mean_squared_error(labels=labels, predictions=logits)
                metrics = {'mse': mse}
                if mode == tf.estimator.ModeKeys.EVAL:
                    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=metrics)
                train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss=loss, global_step=tf.train.get_global_step())
                return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op, eval_metric_ops=metrics)
            else:
                predictions = {'predicts': logits}
                return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

        super(EmbeddingNet, self).__init__(model_fn=_model_fn_, config=run_config)

    def layer_fn(self, features):
        embedding_list = list()
        for param in self.feature_columns:
            name = param['feature_name']
            buckets = param['hash_buckets']
            dim = param['embedding_dim']
            seq = param['sequence']
            feature_type = param['type']
            value = features[name]

            with tf.variable_scope('linear', reuse=tf.AUTO_REUSE):
                variables = tf.get_variable(name=name, shape=[buckets, dim], dtype=tf.float32,
                                           initializer=tf.truncated_normal_initializer(stddev=0.02))

            if feature_type == 'string':
                indices = tf.strings.to_hash_bucket_fast(value, buckets)
                embedding = tf.nn.embedding_lookup(variables, indices)
            elif feature_type == 'int':
                embedding = tf.nn.embedding_lookup(variables, value)
            else:
                continue

            embedding = tf.layers.flatten(embedding)
            for unit in self.hidden_units:
                embedding = tf.layers.dense(embedding, unit, activation=tf.nn.relu)
            embedding = tf.layers.dense(embedding, dim, activation=tf.nn.tanh)
            embedding_list.append(embedding)

        ret = tf.concat(embedding_list, axis=1)
        ret = tf.layers.dense(ret, 32, activation=tf.nn.tanh)
        ret = tf.nn.l2_normalize(ret)
        return ret


