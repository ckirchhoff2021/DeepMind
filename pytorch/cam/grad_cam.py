import cv2
import numpy as np


class GradCAM:
    def __init__(self, net, layer_name):
        self.net = net
        self.layer_name = layer_name
        self.feature_map = None
        self.gradients = None
        self.net.eval()
        self.hook_handlers = []
        self._reigister_hook()

    def _get_feature_hook(self, module, input, output):
        self.feature_map = output
        print('feature shape: {}'.format(output.size()))

    def _get_grads_hook(self, module, input_grads, output_grad):
        self.gradients = output_grad[0]

    def _reigister_hook(self):
        for (name, module) in self.net.named_modules():
            if name == self.layer_name:
                self.hook_handlers.append(module.register_forward_hook(self._get_feature_hook))
                self.hook_handlers.append(module.register_backward_hook(self._get_grads_hook))

    def remove_hook_handles(self):
        for handle in self.hook_handlers:
            handle.remove()

    def __call__(self, inputs, index=None):
        self.net.zero_grad()
        output = self.net(inputs)
        if not index:
            index = np.argmax(output.cpu().data.numpy())
        target = output[0][index]
        target.backward()

        gradient = self.gradients[0].cpu().data.numpy()
        weight = np.mean(gradient, axis=(1, 2))
        feature = self.feature_map[0].cpu().data.numpy()

        cam = feature * weight[:, np.newaxis, np.newaxis]
        cam = np.sum(cam, axis=0)
        cam = np.maximum(cam, 0)

        cam -= np.min(cam)
        cam /= np.max(cam)
        cam = cv2.resize(cam, (224, 224))
        return cam