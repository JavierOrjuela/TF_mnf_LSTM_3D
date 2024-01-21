import tensorflow as tf
from tensorflow.keras.layers import Flatten, MaxPool2D, ReLU, Softmax,MaxPool3D,BatchNormalization
from tensorflow.keras.layers import Input,Activation,ZeroPadding3D,Add,MaxPooling3D,GlobalAveragePooling3D,GlobalMaxPooling3D,Dense,Conv3D,Lambda,Multiply
from tf_mnf.layers import MNFConv2D, MNFDense,MNFConv3D
from classification_models_3D.tfkeras import Classifiers
#from volumentations import *
import functools
import tensorflow.keras.backend as K 
from tensorflow.keras.models import Model
import tensorflow_probability as tfp
import collections
tfd = tfp.distributions
tfb = tfp.bijectors
tfpl = tfp.layers
ModelParams = collections.namedtuple(
    'ModelParams',
    ['model_name', 'repetitions', 'residual_block', 'attention']
)

# def ChannelSE(reduction=16, **kwargs):
#     channels_axis = 4 if backend.image_data_format() == 'channels_last' else 1
#     def layer(input_tensor):
#         # get number of channels/filters
#         channels = backend.int_shape(input_tensor)[channels_axis]
#         x = input_tensor
#         # squeeze and excitation block in PyTorch style with
#         x = layers.GlobalAveragePooling3D()(x)
#         x = layers.Lambda(expand_dims, arguments={'channels_axis': channels_axis})(x)
#         x = layers.Conv3D(channels // reduction, (1, 1, 1), kernel_initializer='he_uniform')(x)
#         x = layers.Activation('relu')(x)
#         x = layers.Conv3D(channels, (1, 1, 1), kernel_initializer='he_uniform')(x)
#         x = layers.Activation('sigmoid')(x)
#         # apply attention
#         x = layers.Multiply()([input_tensor, x])
#         return x
#     return layer
class  ChannelSE(tf.keras.layers.Layer):
    def __init__(self,reduction=16,channel=16):
        super().__init__()
        self.reduction=reduction
        self.channels= channel
        self.avgpool =GlobalAveragePooling3D()
        #self.lambda_ = Lambda(self.expand_dims)
        self.mnfconv3d_1 = Conv3D(self.channels // self.reduction, (1, 1, 1), kernel_initializer='he_uniform')
        self.activation_1 = Activation('relu')
        self.mnfconv3d_2 = Conv3D(self.channels, (1, 1, 1), kernel_initializer='he_uniform')
        self.activation_2 = Activation('sigmoid')
        self.multiply = Multiply()
    def call(self, inputs):  
        x= self.avgpool(inputs)
        x= self.expand_dims(x)
        x= self.mnfconv3d_1(x)
        x= self.activation_1(x)
        x= self.mnfconv3d_2(x)
        x=self.activation_2(x)
        x= self.multiply([inputs, x])
        return x
    
    def expand_dims(self,x):
        #return x[:, None, None, None, :]
        return tf.expand_dims(tf.expand_dims(tf.expand_dims(x,axis=1),axis=2),axis=3)

    def kl_div(self):
        kl_list= [self.mnfconv3d_1,self.mnfconv3d_2]
        
        return sum(lyr.kl_div() for lyr in kl_list if hasattr(lyr, "kl_div"))
        

class  residual_conv_block(tf.keras.layers.Layer):
    def __init__(self,filters, strides=[1,1,1], attention=None, cut='pre'):
        super().__init__()
        self.cut=cut
        self.attention=attention
        self.bn_1 = BatchNormalization()
        self.activation_1 = Activation('relu')
        self.zeropadding_1 = ZeroPadding3D(padding=(1, 1, 1))
        self.mnfconv3d_1 = MNFConv3D(filters, (3, 3, 3), strides=strides)
        self.bn_2 = BatchNormalization()
        self.activation_2 = Activation('relu')
        self.zeropadding_2 = ZeroPadding3D(padding=(1, 1, 1))
        self.mnfconv3d_2 = MNFConv3D(filters, (3, 3, 3))
        self.add = Add()
        if self.cut == 'post':
            self.shortcut  = MNFConv3D(filters, (1,1,1), strides=strides)
        else:
            self.shortcut = None
        if self.attention is not None:
            self.attentionlayer=self.attention(reduction=16,channel=filters)
        else:
            self.attentionlayer=None
    def call(self, inputs):
        x = self.bn_1(inputs)
        x = self.activation_1(x)
        if  self.cut == 'pre':
            shortcut = inputs
        elif  self.cut == 'post':
            shortcut = self.shortcut(x)
        else:
             raise ValueError('Cut type not in ["pre", "post"]')
        x = self.zeropadding_1(x)
        x = self.mnfconv3d_1(x)
        x = self.bn_2(x)
        x = self.activation_2(x)
        x = self.zeropadding_2(x)
        x = self.mnfconv3d_2(x)
        if self.attention is not None:
            x = self.attentionlayer(x)
        x = self.add([x,shortcut])
        return x
    def kl_div(self):
        kl_list= [self.mnfconv3d_1,self.mnfconv3d_2,self.shortcut,self.attentionlayer]
        
        return sum(lyr.kl_div() for lyr in kl_list if hasattr(lyr, "kl_div"))
    
class  residual_bottleneck_block(tf.keras.layers.Layer):
    def __init__(self,filters, strides=[1,1,1], attention=None, cut='pre'):
        super().__init__()
        self.cut=cut
        self.attention=attention
        self.bn_1 = BatchNormalization()
        self.activation_1 = Activation('relu')
        self.mnfconv3d_1 = MNFConv3D(filters, (1, 1, 1))
        self.bn_2 = BatchNormalization()
        self.activation_2 = Activation('relu')
        self.zeropadding_1 = ZeroPadding3D(padding=(1, 1, 1))
        self.mnfconv3d_2 = MNFConv3D(filters, (3, 3, 3),strides=strides)
        self.bn_3 = BatchNormalization()
        self.activation_3 = Activation('relu')
        self.mnfconv3d_3 = MNFConv3D(filters*4, (1, 1, 1))
        self.add = Add()
        if  self.cut == 'post':
            self.shortcut  = MNFConv3D(filters*4, (1, 1, 1), strides=strides)
        else:
            self.shortcut = None
        if self.attention is not None:
            self.attentionlayer=self.attention(reduction=16,channel=filters*4)
        else:
            self.attentionlayer=None
        

    def call(self, inputs):
        x = self.bn_1(inputs)
        x = self.activation_1(x)
        if  self.cut == 'pre':
            shortcut = inputs
        elif  self.cut == 'post':
            shortcut = self.shortcut(x)
        else:
             raise ValueError('Cut type not in ["pre", "post"]')
        x = self.mnfconv3d_1(x)
        x = self.bn_2(x)
        x = self.activation_2(x)
        x = self.zeropadding_1(x)
        x = self.mnfconv3d_2(x)
        x = self.bn_3(x)
        x = self.activation_3(x)
        x = self.mnfconv3d_3(x)
        if self.attention is not None:
            x = self.attentionlayer(x)
        x = self.add([x,shortcut])
        return x
    def kl_div(self):
        kl_list= [self.mnfconv3d_1,self.mnfconv3d_2,self.mnfconv3d_3,self.shortcut,self.attentionlayer]
        
        return sum(lyr.kl_div() for lyr in kl_list if hasattr(lyr, "kl_div"))
    
MODELS_PARAMS = {
    'resnet18': ModelParams('resnet18', (2, 2, 2, 2), residual_conv_block, ChannelSE),
    'resnet34': ModelParams('resnet34', (3, 4, 6, 3), residual_conv_block, None),
    'resnet50': ModelParams('resnet50', (3, 4, 6, 3), residual_bottleneck_block, None),
    'resnet101': ModelParams('resnet101', (3, 4, 23, 3), residual_bottleneck_block, None),
    'resnet152': ModelParams('resnet152', (3, 8, 36, 3), residual_bottleneck_block, None),
    'seresnet18': ModelParams('seresnet18', (2, 2, 2, 2), residual_bottleneck_block, ChannelSE),
    'seresnet34': ModelParams('seresnet34', (3, 4, 6, 3), residual_conv_block, ChannelSE),
}

class MNFLeNet3DResnet(tf.keras.Model):
    def __init__(self, model_name='seresnet18',\
                  pooling='avg',stride_size=2, init_filters=16,repetitions=(2,2,2,2),n_classes=4):
        super().__init__()
        self.n_classes=n_classes
        model_params=MODELS_PARAMS[model_name]
        if type(stride_size) not in (tuple, list):
            stride_size = [
            (stride_size, stride_size, stride_size,),
            (stride_size, stride_size, stride_size,),
            (stride_size, stride_size, stride_size,),
            (stride_size, stride_size, stride_size,),
            (stride_size, stride_size, stride_size,),
                        ]
        else:
            stride_size = list(stride_size)

        if len(stride_size) < 3:
            print('Error: stride_size length must be 3 or more')
            return None

        if len(stride_size) - 1 != len(repetitions):
            print('Error: stride_size length must be equal to repetitions length - 1')
            return None

        for i in range(len(stride_size)):
            if type(stride_size[i]) not in (tuple, list):
                stride_size[i] = (stride_size[i], stride_size[i], stride_size[i])
        self.stride_size=stride_size
        self.bn_1 = BatchNormalization()
        self.dense = MNFDense(tfp.layers.MultivariateNormalTriL.params_size(self.n_classes))
        self.denseprob = tfp.layers.MultivariateNormalTriL(self.n_classes)
        self.zeropadding_1 = ZeroPadding3D(padding=(3, 3, 3))
        self.mnfconv3d_1 = MNFConv3D(init_filters, (7, 7, 7), strides=self.stride_size[0])
        self.bn_2 = BatchNormalization()
        self.activation_1 = Activation('relu')
        self.bn_3 = BatchNormalization()
        self.activation_2 = Activation('relu')
        self.zeropadding_2 = ZeroPadding3D(padding=(1, 1, 1))
        self.pool =(self.stride_size[1][0] + 1, self.stride_size[1][1] + 1, self.stride_size[1][2] + 1)
        self.max = MaxPooling3D(self.pool, strides=self.stride_size[1], padding='valid')
        self.pooling = pooling
        if self.pooling == 'avg':
            self.pool_layer = GlobalAveragePooling3D(name='avg_pool')
        elif self.pooling == 'max':
            self.pool_layer  = GlobalMaxPooling3D(name='max_pool')
        self.dict=collections.defaultdict(dict)
        ResidualBlock = model_params.residual_block
        if model_params.attention:
            Attention = model_params.attention
        else:
            Attention=None
        stride_count = 2
        for stage, rep in enumerate(repetitions):
            for block in range(rep):
                filters = init_filters * (2 ** stage)
                if block == 0 and stage == 0:
                    self.dict[stage][block]= ResidualBlock(filters,strides=[1,1,1],cut='post',attention=Attention)
                elif block == 0:
                    self.dict[stage][block]= ResidualBlock(filters,strides=self.stride_size[stride_count],cut='post',attention=Attention)
                    stride_count += 1
                else:
                    self.dict[stage][block]= ResidualBlock(filters,strides=[1,1,1],cut='pre',attention=Attention)

        for key, value in  self.dict.items():
            for key1, value1 in  value.items():
                setattr(self, '_layer_'+str(key)+str(key1), value1)

    def call(self, inputs):
        x = self.bn_1(inputs)
        x = self.zeropadding_1(x)
        x = self.mnfconv3d_1(x)
        x = self.bn_2(x)
        x = self.activation_1(x)
        x = self.zeropadding_2(x)
        x = self.max(x)
        for key, value in  self.dict.items():
            for key1, _ in  value.items():
                x = getattr(self, '_layer_'+str(key)+str(key1))(x)
        x = self.bn_3(x)
        x = self.activation_2(x)
        x = self.pool_layer(x)
        x= self.dense(x)
        x= self.denseprob(x)
        return x

    def get_all_keys(self, d):
            for _, value in d.items():
                yield value
                if isinstance(value, dict):
                    yield from self.get_all_keys(value)




    def kl_div(self):
        """Compute current KL divergence of the whole model.
        Can be used as a regularization term during training.
        """
        def rgetattr(obj, attr):
            def _getattr(obj, attr):
                return hasattr(obj, attr)
            return functools.reduce(_getattr, [obj] + attr.split('.'))
        
        return sum(lyr.kl_div() for lyr in self.layers if rgetattr(lyr, "kl_div"))
        #return [lyr for lyr in self.layers if rgetattr(lyr, "kl_div")]