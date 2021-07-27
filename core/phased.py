'''
Custom tensorflow keras layers
'''
import tensorflow as tf
from tensorflow import constant_initializer


def _exponential_initializer(min, max, dtype=None):
    def in_func(shape, dtype=dtype):
        initializer = tf.random_uniform_initializer(
                        tf.math.log(min),
                        tf.math.log(max)
                        )
        return tf.math.exp(initializer(shape))
    return in_func

class PhasedLSTM(tf.keras.layers.Layer):
    def __init__(self,
                 units,
                 leak_rate=0.001,
                 ratio_on=0.1,
                 period_init_min=0.1,
                 period_init_max=1000.0,
                 rec_activation = tf.math.sigmoid,
                 out_activation = tf.math.tanh,
                 name='plstm',
                 **kwargs):
        super(PhasedLSTM, self).__init__(name=name)

        self.state_size = [units,units]
        self.output_size = units

        self.units = units
        self._leak = leak_rate
        self._ratio_on = ratio_on
        self._rec_activation = rec_activation
        self._out_activation = out_activation
        self.period_init_min = period_init_min
        self.period_init_max = period_init_max

        self.cell = tf.keras.layers.LSTMCell(units, **kwargs)

    def _get_cycle_ratio(self, time, phase, period):
        """Compute the cycle ratio in the dtype of the time."""
        phase_casted = tf.cast(phase, dtype=time.dtype)
        period_casted = tf.cast(period, dtype=time.dtype)
        time = tf.reshape(time, [tf.shape(time)[0],1]) #This change
        shifted_time = time - phase_casted
        cycle_ratio = (shifted_time%period_casted) / period_casted
        return tf.cast(cycle_ratio, dtype=tf.float32)

    def build(self, input_shape):
        self.period = self.add_weight(
                        name="period",
                        shape=[self.units],
                        initializer=_exponential_initializer(
                                            self.period_init_min,
                                            self.period_init_max),
                        trainable=True)

        self.phase = self.add_weight(name="phase",
                                     shape=[self.units],
                                     initializer=tf.random_uniform_initializer(
                                                         0.0,
                                                         self.period),
                                     trainable=True)
        self.ratio_on = self.add_weight(name="ratio_on",
                                        shape=[self.units],
                                        initializer=constant_initializer(self._ratio_on),
                                        trainable=True)

    def call(self, input, states):
        inputs, times = input[:,1:], input[:,0] #This change

        # =================================
        # CANDIDATE CELL AND HIDDEN STATE
        # =================================
        prev_hs, prev_cs = states
        output, (hs, cs) = self.cell(inputs, states)

        # =================================
        # TIME GATE
        # =================================
        cycle_ratio = self._get_cycle_ratio(times, self.phase, self.period)

        k_up = 2 * cycle_ratio / self.ratio_on
        k_down = 2 - k_up
        k_closed = self._leak * cycle_ratio

        k = tf.where(cycle_ratio < self.ratio_on, k_down, k_closed)
        k = tf.where(cycle_ratio < 0.5 * self.ratio_on, k_up, k)

        # =================================
        # UPDATE STATE USING TIME GATE VALUES
        # =================================
        new_h = k * hs + (1 - k) * prev_hs
        new_c = k * cs + (1 - k) * prev_cs

        return new_h, (new_h, new_c)
