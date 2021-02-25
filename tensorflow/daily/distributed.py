import os
import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_integer("task_index", None, "Worker task index")
flags.DEFINE_string("ps_hosts", "p0","ps hosts")
flags.DEFINE_string("worker_hosts", "w0,w1", "worker hosts")
flags.DEFINE_string("job_name", None, "job name: worker or ps")

FLAGS = tf.app.flags.FLAGS

def model(x):
    y1 = tf.layers.dense(x, 200, activation=tf.nn.relu)
    y2 = tf.layers.dense(y1, 100, activation=tf.nn.relu)
    y3 = tf.layers.dense(y2, 10)
    return y3

def main(unused_argv):
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)

    datas = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    datas = datas.shuffle(buffer_size=1000).batch(64, drop_remainder=False).repeat(5)
    iterator = datas.make_one_shot_iterator()
    element = iterator.get_next()

    print("job name = %s" % FLAGS.job_name)
    print("task index = %d" % FLAGS.task_index)
    is_chief = FLAGS.task_index == 0

    ps_spec = FLAGS.ps_hosts.split(",")
    worker_spec = FLAGS.worker_hosts.split(",")
    cluster = tf.train.ClusterSpec({"ps": ps_spec, "worker": worker_spec})
    worker_count=len(worker_spec)

    server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)
    # join the ps server
    if FLAGS.job_name == "ps":
        server.join()

    with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/task:%d" % (FLAGS.task_index),
                                                  cluster=cluster)):
        images = tf.placeholder(tf.float32, [None, 784])
        labels = tf.placeholder(tf.int32, [None, 10])

        logits = model(images)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

        hooks = [tf.train.StopAtStepHook(last_step=2000)]
        global_step = tf.train.get_or_create_global_step()
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-04)

        if FLAGS.is_sync:
            optimizer = tf.train.SyncReplicasOptimizer(optimizer, replicas_to_aggregate=FLAGS.num_workers,
                                                       total_num_replicas=FLAGS.num_workers)
            hooks.append(optimizer.make_session_run_hook((FLAGS.task_index == 0)))

        train_op = optimizer.minimize(loss, global_step=global_step,
                                      aggregation_method=tf.AggregationMethod.ADD_N)

        with tf.train.MonitoredTrainingSession(master=server.target,
                                               is_chief=(FLAGS.task_index == 0),
                                               checkpoint_dir=os.path.join('../../output', 'distributed'),
                                               hooks=hooks) as mon_sess:

            while not mon_sess.should_stop():
                img_batch, label_batch = element
                _, ls, step = mon_sess.run([train_op, loss, global_step],
                                           feed_dict={images: img_batch, labels: label_batch})
                if step % 100 == 0:
                    print("Train step %d, loss: %f" % (step, ls))


if __name__=="__main__":
    tf.app.run(main)
