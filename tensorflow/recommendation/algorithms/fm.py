import tensorflow as tf

'''
input_config = {
    'user':[
        {
            'feature_name': 'xxx',
            'feature_dim': 10,
            'feature_type': 'string/float/int',
            'feature_seq': 1,
            'hash_bucket_num': 20
        },
        ...
    ],
    'item':[
        {
            'feature_name': 'xxx',
            'feature_dim': 10,
            'feature_type': 'string/float/int',
            'feature_seq': 1,
            'hash_bucket_num': 20
        }
    ]
}

net_config = {
    'user_weights': {
        'names': ['user_w1','user_b1','user_w2','user_b2'],
        'shapes':[64, 32]

    },
    'item_weights': {
        'names':  ['item_w1','item_b1','item_w2','item_b2'],
        'shapes': [64, 32]
    }
}
'''

class FMModel:
    @staticmethod
    def get_variable(variable_name, variable_shape):
        variable = tf.get_variable(variable_name, shape=variable_shape, dtype=tf.float32,
                                   initializer=tf.truncated_normal_initializer(stddev=0.02))
        return variable

    def __init__(self, input_config, net_config):
        self.user_dims = 0
        self.item_dims = 0
        self.variable_dict = dict()
        self.placeholder_dict = dict()
        self.user_features = input_config['user']
        print('==> user features count: ', len(user_features))
        for feature in self.user_features:
            feature_name = feature['feature_name']
            feature_dim = feature['feature_dim']
            feature_type = feature['feature_type']
            feature_seq = feature['feature_seq']
            bucket_num = feature['hash_bucket_num']
            variable = self.get_variable(feature_name, shape=[bucket_num, feature_dim])
            self.variable_dict[feature_name] = variable
            self.placeholder_dict[feature_name] = tf.placeholder(tf.float32, [None, feature_seq])
            self.user_dims += feature_dim * feature_seq

        self.item_dims = 0
        self.item_features = input_config['item']
        print('==> item features count: ', len(item_features))
        for feature in self.item_features:
            feature_name = feature['feature_name']
            feature_dim = feature['feature_dim']
            feature_type = feature['feature_type']
            feature_seq = feature['feature_seq']
            bucket_num = feature['hash_bucket_num']
            variable = tf.get_variable(feature_name, shape=[bucket_num, feature_dim], dtype=tf.float32,
                                       initializer=tf.truncated_normal_initializer(stddev=0.02))
            self.variable_dict[feature_name] = variable
            self.placeholder_dict[feature_name] = tf.placeholder(tf.float32, [None, feature_seq])
            self.item_dims += feature_dim * feature_seq


        self.user_weights = net_config['user_weights']
        ku = self.user_dims
        for i in range(self.user_weights['shapes']):
            n = self.user_weights['shapes'][i]
            weight_name = self.user_weights['names'][i]
            bias_name = self.user_weights['names'][i+1]
            w = tf.get_variable(weight_name, shape=[ku, n], dtype=tf.float32,
                                initializer=tf.truncated_normal_initializer(stddev=0.02))
            self.variable_dict[weight_name] = w
            b = tf.get_variable(weight_name, shape=[n], dtype=tf.float32,
                                initializer=tf.truncated_normal_initializer(stddev=0.02))
            self.variable_dict[bias_name] = b
            ku = n

        self.item_weights = net_config['item_weights']
        ki = self.item_dims
        for i in range(self.item_weights['shapes']):
            n = item_weights['shapes'][i]
            weight_name = self.item_weights['names'][i]
            bias_name = self.item_weights['names'][i + 1]
            w = tf.get_variable(weight_name, shape=[ki, n], dtype=tf.float32,
                                initializer=tf.truncated_normal_initializer(stddev=0.02))
            self.variable_dict[weight_name] = w
            b = tf.get_variable(weight_name, shape=[n], dtype=tf.float32,
                                initializer=tf.truncated_normal_initializer(stddev=0.02))
            self.variable_dict[bias_name] = b
            ki = n



    def get_input_tensor(self, input_feature):
        vec = list()
        for feature in input_feature:
            feature_name = feature['feature_name']
            feature_type = feature['feature_type']
            feature_seq = feature['feature_seq']
            bucket_num = feature['hash_bucket_num']
            varibles = self.variable_dict[feature_name]
            placeholder = self.placeholder_dict[feature_name]
            if feature_type == 'string':
                indices = tf.strings.to_hash_bucket_fast(placeholder, bucket_num)
                embeddings = tf.nn.embedding_lookup(varibles, indices)
                embeddings = tf.layers.flatten(embeddings)
            elif feature_type == 'int':
                embeddings = tf.nn.embedding_lookup(placeholder, indices)
                embeddings = tf.layers.flatten(embeddings)
            else:
                embeddings = tf.layers.flatten(placeholder)
            vec.append(embeddings)
        vec = tf.concat(vec, axis=1)
        return vec

    def get_embedding(self, input_feature, weights_names):
        count = len(weights_names)
        vec = self.get_input_tensor(input_feature)
        for i in range(0, count, 2):
            weight_name = weights_names[i]
            bias_name = weights_names[i+1]
            w = self.variable_dict[weight_name]
            b = self.variable_dict[bias_name]
            vec = tf.matmul(vec, w) + b
            if i < count -2:
                vec = tf.nn.relu(vec)
        vec = tf.nn.l2_normalize(vec)
        return vec

    def get_feed_dict(self, input_info):
        feed_dict = dict()
        for feature_name in input_info.keys():
            placeholder = self.placeholder_dict[feature_name]
            feed_dict[placeholder] = input_info[feature_name]
        return feed_dict


    def train(self, user_info, positive_info, negative_info):
        init_op = [tf.global_variables_initializer(), tf.local_variables_initializer()]
        u_feed_dict = self.get_feed_dict(user_info)
        p_feed_dict = self.get_feed_dict(positive_info)
        n_feed_dict = self.get_feed_dict(negative_info)

        u_embedding = self.get_embedding(self.user_features, self.user_weights['names'])
        i_embedding = self.get_embedding(self.item_features, self.item_weights['names'])

        with tf.Session() as sess:
            sess.run(init_op)
            u_vec = sess.run(u_embedding, feed_dict=u_feed_dict)
            p_vec = sess.run(i_embedding, feed_dict=p_feed_dict)
            n_vec = sess.run(i_embedding, feed_dict=n_feed_dict)



                    
       



