import tensorflow as tf


def build_layers(feature_columns, hidden_units):
    variable_dict = dict()
    for param in feature_columns:
        name = param['feature_name']
        dim = param['embedding_dim']
        buckets = param['hash_buckets']
        with tf.variable_scope('linear', reuse=tf.AUTO_REUSE):
            variable = tf.get_variable(name=name, shape=[buckets, dim], dtype=tf.float32,
                                       initializer=tf.truncated_normal_initializer(stddev=0.02))
            variable_dict[name] = variable
        
    def layer_fn(features):
        embedding_list = list()
        for param in feature_columns:
            name = param['feature_name']
            buckets = param['hash_buckets']
            dim = param['embedding_dim']
            seq = param['sequence']
            feature_type = param['type']
            value = features[name]

            if feature_type == 'string':
                indices = tf.strings.to_hash_bucket_fast(value, buckets)
                embedding = tf.nn.embedding_lookup(variable_dict[name], indices)
            elif feature_type == 'int':
                embedding = tf.nn.embedding_lookup(variable_dict[name], value)
            else:
                continue
            # tf.add_to_collection(tf.GraphKeys.WEIGHTS, embedding)
            embedding = tf.layers.flatten(embedding)
            for unit in hidden_units:
                embedding = tf.layers.dense(embedding, unit, activation=tf.nn.relu)
            embedding = tf.layers.dense(embedding, dim, activation=tf.nn.tanh)
            embedding_list.append(embedding)

        ret = tf.concat(embedding_list, axis=1)
        ret = tf.layers.dense(ret, 32, activation=tf.nn.tanh)
        ret = tf.nn.l2_normalize(ret)
        return ret

    return layer_fn



            
        
        
