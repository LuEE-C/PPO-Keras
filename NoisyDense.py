from keras import backend as K
from keras.engine.topology import Layer
from keras import activations, initializers, regularizers, constraints
import numpy as np

class NoisyDense(Layer):

    def __init__(self, units,
                 sigma_init=0.02,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(NoisyDense, self).__init__(**kwargs)
        self.units = units
        self.sigma_init = sigma_init
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

    def build(self, input_shape):
        assert len(input_shape) >= 2
        self.input_dim = input_shape[-1]

        self.kernel = self.add_weight(shape=(self.input_dim, self.units),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)

        self.sigma_kernel = self.add_weight(shape=(self.input_dim, self.units),
                                      initializer=initializers.Constant(value=self.sigma_init),
                                      name='sigma_kernel'
                                      )
        self.epsilon_kernel = self.add_weight(shape=(self.input_dim, self.units),
                                              initializer=self.kernel_initializer,
                                              trainable=False,
                                              name='epsilon_kernel')
        self.epsilon_bias = self.add_weight(shape=(self.units,),
                                            initializer=self.kernel_initializer,
                                            trainable=False,
                                            name='epsilon_bias')

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
            self.sigma_bias = self.add_weight(shape=(self.units,),
                                        initializer=initializers.Constant(value=self.sigma_init),
                                        name='sigma_bias')
        else:
            self.bias = None
            self.epsilon_bias = None
        # self.sample_noise()
        super(NoisyDense, self).build(input_shape)


    def call(self, X):
        perturbation = self.sigma_kernel * self.epsilon_kernel
        perturbed_kernel = self.kernel + perturbation
        output = K.dot(X, perturbed_kernel)
        if self.use_bias:
            bias_perturbation = self.sigma_bias * self.epsilon_bias
            perturbed_bias = self.bias + bias_perturbation
            output = K.bias_add(output, perturbed_bias)
        if self.activation is not None:
            output = self.activation(output)
        return output

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)

    def sample_noise(self):
        K.set_value(self.epsilon_kernel, np.random.normal(loc=0, scale=1, size=(self.input_dim, self.units)))
        K.set_value(self.epsilon_bias, np.random.normal(loc=0, scale=1, size=(self.units,)))

    def remove_noise(self):
        K.set_value(self.epsilon_kernel, np.zeros(shape=(self.input_dim, self.units)))
        K.set_value(self.epsilon_bias, np.zeros(shape=(self.units,)))

    def get_config(self):
        config = {
            'units': self.units,
            'sigma_init': self.sigma_init,
            'sigma_kernel': self.sigma_kernel,
            'sigma_bias': self.sigma_bias,
            'epsilon_bias': self.epsilon_bias,
            'epsilon_kernel': self.epsilon_kernel,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(NoisyDense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
