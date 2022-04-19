#coding=utf-8
import tensorflow as tf
import tensorflow.keras as keras


class DepthWiseConv(keras.Model):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(DepthWiseConv, self).__init__()
        self.conv1 = keras.layers.Conv2D(filters=in_channels, kernel_size=kernel_size, strides=stride, padding='same', groups=in_channels, use_bias=False)
        self.relu = keras.layers.ReLU()
        self.bn1 = keras.layers.BatchNormalization()
        self.conv2 = keras.layers.Conv2D(filters=out_channels, kernel_size=1, strides=1, padding='valid', use_bias=False)
        self.bn2 = keras.layers.BatchNormalization()

    def call(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.bn2(y)
        y = self.relu(y)
        return y


def predict_flow():
    return keras.layers.Conv2D(filters=2, kernel_size=3, strides=1, padding='same', use_bias=False)


class DepthwiseDeconv(keras.Model):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(DepthwiseDeconv, self).__init__()
        self.upsample = keras.layers.UpSampling2D(size=(2,2))
        self.conv1 = keras.layers.Conv2D(filters=in_channels, kernel_size=kernel_size, strides=stride, padding='same',groups=in_channels, use_bias=False)
        self.conv2 = keras.layers.Conv2D(filters=out_channels, kernel_size=1, strides=1, padding='valid', use_bias=False)
        self.relu = keras.layers.ReLU()
        self.bn1 = keras.layers.BatchNormalization()
        self.bn2 = keras.layers.BatchNormalization()

    def call(self, x):
        y = self.upsample(x)
        y = self.relu(self.bn1(self.conv1(y)))
        y = self.relu(self.bn2(self.conv2(y)))
        return y



class DepthWiseFlowNetS(keras.Model):
    def __init__(self):
        super(DepthWiseFlowNetS,self).__init__()
        self.conv1   = DepthWiseConv(6,    64,    kernel_size=7, stride=2)
        self.conv2   = DepthWiseConv(64,   128,   kernel_size=5, stride=2)
        self.conv3   = DepthWiseConv(128,  256,   kernel_size=5, stride=2)
        self.conv3_1 = DepthWiseConv(256,  256,   kernel_size=3, stride=1)
        self.conv4   = DepthWiseConv(256,  512,   kernel_size=3, stride=2)
        self.conv4_1 = DepthWiseConv(512,  512,   kernel_size=3, stride=1)
        self.conv5   = DepthWiseConv(512,  512,   kernel_size=3, stride=2)
        self.conv5_1 = DepthWiseConv(512,  512,   kernel_size=3, stride=1)
        self.conv6   = DepthWiseConv(512,  1024,  kernel_size=3, stride=2)
        self.conv6_1 = DepthWiseConv(1024, 1024,  kernel_size=3, stride=1)

        self.deconv5 = DepthwiseDeconv(1024, 512)
        self.deconv4 = DepthwiseDeconv(1026, 256)
        self.deconv3 = DepthwiseDeconv(770,  128)
        self.deconv2 = DepthwiseDeconv(386,  64)

        self.predict_flow6 = predict_flow()
        self.predict_flow5 = predict_flow()
        self.predict_flow4 = predict_flow()
        self.predict_flow3 = predict_flow()
        self.predict_flow2 = predict_flow()

        self.upsample = keras.layers.UpSampling2D(size=(2,2))
    

    def call(self, x):
        out_conv2 = self.conv2(self.conv1(x))
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6_1(self.conv6(out_conv5))

        flow6       = self.predict_flow6(out_conv6)
        flow6_up    = self.upsample(flow6)
        out_deconv5 = self.deconv5(out_conv6)

        concat5     = tf.concat([out_conv5,out_deconv5,flow6_up], -1)
        flow5       = self.predict_flow5(concat5)
        flow5_up    = self.upsample(flow5)
        out_deconv4 = self.deconv4(concat5)

        concat4     = tf.concat([out_conv4,out_deconv4,flow5_up], -1)
        flow4       = self.predict_flow4(concat4)
        flow4_up    = self.upsample(flow4)
        out_deconv3 = self.deconv3(concat4)

        concat3     = tf.concat([out_conv3,out_deconv3,flow4_up], -1)
        flow3       = self.predict_flow3(concat3)
        flow3_up    = self.upsample(flow3)
        out_deconv2 = self.deconv2(concat3)

        concat2     = tf.concat([out_conv2,out_deconv2,flow3_up], -1)
        flow2       = self.predict_flow2(concat2)

        return flow2




class FlowMotionClassify(keras.Model):
    def __init__(self, n_label=4):
        super(FlowMotionClassify, self).__init__()
        self.conv1 = keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same', use_bias=False)
        self.conv2 = keras.models.Sequential([
            keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same', use_bias=False),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU()
        ])

        self.max_pool = keras.layers.MaxPool2D((2, 2))
        self.conv3 = keras.models.Sequential([
            keras.layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='same', use_bias=True),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),

            keras.layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='same', use_bias=True),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
        ])
        
        self.conv3_1 = keras.layers.Conv2D(filters=128, kernel_size=1, strides=1, padding='valid', use_bias=True)
    
        self.conv4 = keras.models.Sequential([
            keras.layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='same', use_bias=True),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),

            keras.layers.Conv2D(filters=256, kernel_size=3, strides=1, padding='same', use_bias=True),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
        ])
        
        self.conv4_1 = keras.layers.Conv2D(filters=256, kernel_size=1, strides=1, padding='valid', use_bias=True)
        self.conv5 = keras.models.Sequential([
            keras.layers.Conv2D(filters=256, kernel_size=3, strides=1, padding='same', use_bias=True),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU()
        ])

        self.avg_pool = keras.layers.AveragePooling2D((4,4))
        self.fc = keras.models.Sequential([
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(n_label),
        ])

    def call(self, x):
        y1 = self.conv1(x)
        y2 = self.conv2(y1)
        y2 = self.max_pool(y2)
        
        y3 = self.conv3(y2)
        y3_1 = self.conv3_1(y2)
        y3 = y3 + y3_1
        y3 = self.max_pool(y3)

        y4 = self.conv4(y3)
        y4_1 = self.conv4_1(y3)
        y4 = y4 + y4_1
        y4 = self.max_pool(y4)

        y5 = self.conv5(y4)
        p = self.avg_pool(y5)
        p = tf.reshape(p, [-1, 3072])
        y = self.fc(p)
        return y



class OpticalFlowMotion(keras.Model):
    def __init__(self,  n_label=4):
        super(OpticalFlowMotion, self).__init__()
        self.n_label = n_label
        self.flownet = DepthWiseFlowNetS()
        self.classify = FlowMotionClassify(n_label=self.n_label)

    def call(self, x):
        flow = self.flownet(x)
        y = self.classify(flow)
        return y




if __name__ == '__main__':
    '''
    net = DepthWiseFlowNetS()
    net.build(input_shape=(1,512,384,6))
    net.summary()
    '''

    '''
    net = FlowMotionClassify()
    net.build(input_shape=(1,128,96,2))
    net.summary()
    '''

    net = OpticalFlowMotion()
    net.build(input_shape=(1,512,384,6))
    net.summary()
