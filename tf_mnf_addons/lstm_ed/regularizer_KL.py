import trans_finder.cta.lstm_ed.generated_random_variables as generated_random_variables
import trans_finder.cta.lstm_ed.random_variable as random_variable
import tensorflow as tf

class NormalKLDivergence(tf.keras.regularizers.Regularizer):
  """KL divergence regularizer from an input to the normal distribution."""

  def __init__(self, mean=0., stddev=1., scale_factor=1.):
    """Constructs regularizer where default is a KL towards the std normal."""
    self.mean = mean
    self.stddev = stddev
    self.scale_factor = scale_factor

  def __call__(self, x):
    """Computes regularization given an input ed.RandomVariable."""
    
    prior = generated_random_variables.Independent(
        generated_random_variables.Normal(
            loc=tf.broadcast_to(self.mean, x.distribution.event_shape),
            scale=tf.broadcast_to(self.stddev, x.distribution.event_shape)
        ).distribution,
        reinterpreted_batch_ndims=len(x.distribution.event_shape))
    regularization = x.distribution.kl_divergence(prior.distribution)
    return self.scale_factor * regularization

  def get_config(self):
    return {
        'mean': self.mean,
        'stddev': self.stddev,
        'scale_factor': self.scale_factor,
    }
def deserialize(config, custom_objects=None):
  return tf.keras.utils.legacy.deserialize_keras_object(
      config,
      module_objects=globals(),
      custom_objects=custom_objects,
      printable_module_name='regularizers',
  )


def get(identifier, value=None):
  """Getter for loading from strings; falls back to Keras as needed."""
  if value is None:
    value = identifier
  if identifier in (None, ''):
    return None
  elif isinstance(identifier, dict):
    try:
      return deserialize(identifier)
    except ValueError:
      pass
  elif isinstance(identifier, str):
    config = {'class_name': str(identifier), 'config': {}}
    try:
      return deserialize(config)
    except ValueError:
      pass
  elif callable(identifier):
    return identifier
  return tf.keras.regularizers.get(value)