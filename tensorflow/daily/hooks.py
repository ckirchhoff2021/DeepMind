import collections
import tensorflow as tf




class LogviewTrainHook(tf.train.SessionRunHook):
    def __init__(self, _metric_ops, _step_op):
        self._step_op = _step_op
        self._metric_ops = {}
        for k, v in _metric_ops.items():
            self._metric_ops[k] = v
        self.cnt = 0

    def before_run(self, run_context):
        tensors = [self._metric_ops, self._step_op]
        return tf.train.SessionRunArgs(tensors)

    def after_run(self, run_context, run_values):
        self.cnt += 1
        if self.cnt % 10 == 0:
            _metric_values, _step = run_values.results
            for k, v in _metric_values.items():
                print('--- {0} is {1} at {2}'.format(k, v, _step))


class EarlyStopping(tf.train.SessionRunHook):
    def __init__(self,_metric_ops, _step_op):
        self._step_op = _step_op
        self._metric_ops = {}
        for k, v in _metric_ops.items():
            self._metric_ops[k] = v
        self.cnt = 0

    def before_run(self, run_context):
        tensors = [self._metric_ops, self._step_op]
        return tf.train.SessionRunArgs(tensors)

    def after_run(self, run_context, run_values):
        _metric_values, _step = run_values.results
        if _step > 20 or _metric_values['accuracy'] > 0.9:
            run_context.request_stop()
            print("REQUESTED_STOP")
            raise ValueError('Model Stopping from EarlyStopping hook')
