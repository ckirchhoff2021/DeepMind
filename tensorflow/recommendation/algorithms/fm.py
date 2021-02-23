import tensorflow as tf
from common_path import *

'''
input_config = {
    'user':[
        {
            'name': 'xxx',
            'dim': 10,
            'type': 'string/float/int',
            'sequence': 1,
            'buckets': 20
        },
        ...
    ],
    'item':[
        {
            'name': 'xxx',
            'dim': 10,
            'type': 'string/float/int',
            'sequence': 1,
            'buckets': 20
        }
    ]
}

'''

class FMModel:
    '''
    twin towers model
    '''
    @staticmethod
    def get_variable(variable_name, variable_shape):
        variable = tf.get_variable(variable_name, shape=variable_shape, dtype=tf.float32,
                                   initializer=tf.truncated_normal_initializer(stddev=0.02))
        return variable

    def _init_model_layers_(self):
        for i, name in enumerate(self.weights_names):
            variable = self.get_variable(name, self.weights_shapes[i])
            self.variable_dict[name] = variable

    @staticmethod
    def _get_placeholder_type_(fea_type):
        if fea_type == 'string':
            return tf.string
        elif fea_type == 'int':
            return tf.int32
        else:
            return tf.float32

    def __init__(self, user_config, item_config):
        self.variable_dict = dict()
        self.placeholder_dict = {
            'user': dict(),
            'positive': dict(),
            'negative': dict()
        }
        self.user_config = user_config
        self.item_config = item_config

        self.user_dims = 0
        print('==> user features count: ', len(self.user_config))
        for feature in self.user_config:
            fea_name = feature['name']
            fea_dim = feature['dim']
            fea_type = feature['type']
            fea_seq = feature['sequence']
            fea_buckets = feature['buckets']

            variable = self.get_variable(fea_name, [fea_buckets, fea_dim])
            self.variable_dict[fea_name] = variable
            placeholder_type = self._get_placeholder_type_(fea_type)
            placeholder_shape = [None, fea_seq] if fea_seq > 1 else [None]
            self.placeholder_dict['user'][fea_name] = tf.placeholder(placeholder_type, placeholder_shape)
            self.user_dims += fea_dim * fea_seq

        self.item_dims = 0
        print('==> item features count: ', len(self.item_config))
        for feature in self.item_config:
            fea_name = feature['name']
            fea_dim = feature['dim']
            fea_type = feature['type']
            fea_seq = feature['sequence']
            fea_buckets = feature['buckets']

            variable = self.get_variable(fea_name, [fea_buckets, fea_dim])
            self.variable_dict[fea_name] = variable
            placeholder_type = self._get_placeholder_type_(fea_type)
            placeholder_shape = [None, fea_seq] if fea_seq > 1 else [None]
            self.placeholder_dict['positive'][fea_name] = tf.placeholder(placeholder_type, placeholder_shape)
            self.placeholder_dict['negative'][fea_name] = tf.placeholder(placeholder_type, placeholder_shape)
            self.item_dims += fea_dim * fea_seq

        self.weights_names = ['user_w1', 'user_b1', 'user_w2', 'user_b2', 'item_w1', 'item_b1', 'item_w2', 'item_b2']
        self.weights_shapes = [(self.user_dims, 64), (64), (64, 32), (32), (self.item_dims, 64), (64), (64, 32), (32)]

        self._init_model_layers_()
        self.sess = tf.Session()
        self.saver = tf.train.Saver()

    def get_tensor(self, input_type='user'):
        config_dict = self.user_config if input_type == 'user' else self.item_config
        placeholder_dict = self.placeholder_dict[input_type]

        vec = list()
        for feature in config_dict:
            fea_name = feature['name']
            fea_dim = feature['dim']
            fea_type = feature['type']
            fea_seq = feature['sequence']
            fea_buckets = feature['buckets']

            varibles = self.variable_dict[fea_name]
            placeholder = placeholder_dict[fea_name]
            if fea_type == 'string':
                indices = tf.strings.to_hash_bucket_fast(placeholder, fea_buckets)
                embeddings = tf.nn.embedding_lookup(varibles, indices)
                embeddings = tf.layers.flatten(embeddings)
            elif fea_type == 'int':
                embeddings = tf.nn.embedding_lookup(varibles, placeholder)
                embeddings = tf.layers.flatten(embeddings)
            else:
                embeddings = tf.layers.flatten(placeholder)
            vec.append(embeddings)

        vec = tf.concat(vec, axis=1)
        return vec

    def get_embedding(self, input_type='user'):
        x = self.get_tensor(input_type)
        w1_names = ['user_w1', 'user_b1', 'user_w2', 'user_b2']
        w2_names = ['item_w1', 'item_b1', 'item_w2', 'item_b2']
        weights_names = w1_names if input_type == 'user' else w2_names
        w1 = self.variable_dict[weights_names[0]]
        b1 = self.variable_dict[weights_names[1]]
        w2 = self.variable_dict[weights_names[2]]
        b2 = self.variable_dict[weights_names[3]]

        y1 = tf.nn.relu(tf.matmul(x, w1) + b1)
        y2 = tf.matmul(y1, w2) + b2
        y2 = tf.nn.l2_normalize(y2)
        return y2

    def get_feed_dict(self, user_info, pos_info, neg_info):
        feed_dict = dict()
        for fea_name in user_info.keys():
            placeholder = self.placeholder_dict['user'][fea_name]
            feed_dict[placeholder] = user_info[fea_name]

        for fea_name in pos_info.keys():
            placeholder = self.placeholder_dict['positive'][fea_name]
            feed_dict[placeholder] = pos_info[fea_name]

        for fea_name in neg_info.keys():
            placeholder = self.placeholder_dict['negative'][fea_name]
            feed_dict[placeholder] = neg_info[fea_name]

        return feed_dict

    def triplet_loss(self, margin=1.0):
        user_vec = self.get_tensor(input_type='user')
        pos_vec = self.get_tensor(input_type='positive')
        neg_vec = self.get_tensor(input_type='negative')
        
        user_embedding = self.get_embedding(input_type='user')
        pos_embedding = self.get_embedding(input_type='positive')
        neg_embedding = self.get_embedding(input_type='negative')

        d1 = tf.reduce_sum(user_embedding * pos_embedding, 1)
        d2 = tf.reduce_sum(user_embedding * neg_embedding, 1)
        loss = tf.maximum(0.0, margin + d1 - d2)
        loss = tf.reduce_mean(loss)
        return loss

    def train(self, user_info, pos_info, neg_info, model_dir, summary_dir):
        feed_dict = self.get_feed_dict(user_info, pos_info, neg_info)
        loss = self.triplet_loss()
        tf.summary.scalar('loss', loss)
        summaries = tf.summary.merge_all()
        optimizer = tf.train.AdamOptimizer(learning_rate=0.005).minimize(loss)
        init_op = [tf.global_variables_initializer(), tf.local_variables_initializer()]
        summary_writer = tf.summary.FileWriter(summary_dir, self.sess.graph)
        self.sess.run(init_op)
        for epoch in range(10):
            _, vloss, summary = self.sess.run([optimizer, loss, summaries], feed_dict=feed_dict)
            summary_writer.add_summary(summary, global_step=epoch)
            print('==> epoch: %d, loss = %f'% (epoch, vloss))
            self.saver.save(self.sess, os.path.join(model_dir, 'fm.ckpt'))


    def batch_train(self, user_info, pos_info, neg_info, model_dir, summary_dir, batch_size=128, epochs=10):
        dataset = tf.data.Dataset.from_tensor_slices((user_info, pos_info, neg_info))
        dataset = dataset.shuffle(buffer_size=1000).batch(batch_size).repeat(epochs)
        iterator = dataset.make_one_shot_iterator()
        element = iterator.get_next()

        loss = self.triplet_loss()
        tf.summary.scalar('loss', loss)
        summaries = tf.summary.merge_all()
        optimizer = tf.train.AdamOptimizer(learning_rate=0.005).minimize(loss)
        init_op = [tf.global_variables_initializer(), tf.local_variables_initializer()]
        summary_writer = tf.summary.FileWriter(summary_dir, self.sess.graph)
        self.sess.run(init_op)
        count = 0

        try:
            while True:
                count += 1
                batch_datas = self.sess.run(element)
                user_data, pos_data, neg_data = batch_datas
                feed_dict = self.get_feed_dict(user_data, pos_data, neg_data)
                _, vloss, summary = self.sess.run([optimizer, loss, summaries], feed_dict=feed_dict)
                summary_writer.add_summary(summary, global_step=count)
                print('==> count: %d, loss = %f' % (count, vloss))
                if count % 10 == 0:
                    self.saver.save(self.sess, os.path.join(model_dir, 'fm.ckpt'), global_step=count)

        except tf.errors.OutOfRangeError:
                print('end!')


    def restore(self, model_dir):
        model_file = tf.train.latest_checkpoint(model_dir)
        self.saver.restore(self.sess, model_file)

    def predict(self, input_info, input_type='user'):
        vec = self.get_embedding(input_type)
        placholder_dict = self.placeholder_dict[input_type]
        feed_dict = dict()
        for fea_name in input_info.keys():
            placeholder = placholder_dict[fea_name]
            feed_dict[placeholder] = input_info[fea_name]
        vec = self.sess.run(vec, feed_dict=feed_dict)
        return vec



def main():
    user_config = [
        {
            'name': 'x1',
            'dim': 10,
            'type': 'string',
            'sequence': 1,
            'buckets': 20
        },
        {
            'name': 'x2',
            'dim': 10,
            'type': 'string',
            'sequence': 3,
            'buckets': 20
        },
        {
            'name': 'x3',
            'dim': 10,
            'type': 'string',
            'sequence': 1,
            'buckets': 20
        }
    ]

    item_config = [
        {
            'name': 'y1',
            'dim': 10,
            'type': 'string',
            'sequence': 1,
            'buckets': 20
        }
    ]

    model = FMModel(user_config, item_config)
    user_info = {
        'x1': ['k1', 'k2', 'k3'],
        'x2': [['v1','v2', 'v3'], ['s1', 's2', 's3'], ['w1','w2', 'w3']],
        'x3': ['n1', 'n2', 'n3']
    }
    pos_info = {
        'y1': ['b1', 'b2', 'b3']
    }
    neg_info = {
        'y1': ['r1', 'r2', 'r3']
    }
    model_dir = os.path.join(output_path, 'fm')
    summary_dir = os.path.join(output_path,'summary')
    # model.train(user_info, pos_info, neg_info, model_dir, summary_dir)
    model.batch_train(user_info, pos_info, neg_info, model_dir, summary_dir, batch_size=2, epochs=10)
    # model.restore(model_dir)
    # embedding = model.predict(pos_info, 'positive')
    # print(embedding)


if __name__ == '__main__':
    main()

       



