# models/networks.py

from tradingbot import cfg, get_logger

_LOG = get_logger("models.networks")

# ------------------------------------------------------------------- #
#  Helpers – cfg look-ups with fallback
# ------------------------------------------------------------------- #

def _hp(section: str, attr: str, default):
    """Shortcut: cfg.<section>.<attr> or fallback to *default*."""
    try:
        return getattr(getattr(cfg, section), attr)
    except AttributeError:
        return default

# size / reg defaults (override in your YAML)
D_HIDDEN_1 = _hp("models", "dense_hidden_1", 128)
D_HIDDEN_2 = _hp("models", "dense_hidden_2", 54)
REG        = l2(_hp("models", "l2", 1e-5))

@register_keras_serializable()
def expand_dims_channel(x):
    from tensorflow.keras.regularizers import l2
    from tensorflow.keras.utils import register_keras_serializable
    import tensorflow as tf
    from tensorflow.keras import Input, Model
    from tensorflow.keras.layers import (
    Dense, LSTM, LayerNormalization, Concatenate, Flatten, Lambda, ConvLSTM2D
    )
    return tf.expand_dims(x, axis=-1)

def get_input_layers(window_size, num_shared_features, num_tradeable, num_ticker_features):
    import tensorflow as tf
    from tensorflow.keras import Input, Model
    from tensorflow.keras.layers import (
    Dense, LSTM, LayerNormalization, Concatenate, Flatten, Lambda, ConvLSTM2D
    )
    from tensorflow.keras.regularizers import l2
    from tensorflow.keras.utils import register_keras_serializable
    """Helper function to create input layers for the models."""
    return {
        'shared_features': Input(shape=(window_size, num_shared_features), name='shared_features_input', dtype=tf.float32),
        'ticker_features': Input(shape=(window_size, num_tradeable, num_ticker_features), name='ticker_features_input', dtype=tf.float32),
        'portfolio_holdings': Input(shape=(num_tradeable,), name='portfolio_holdings_input', dtype=tf.float32),
        'cash_balance': Input(shape=(1,), name='cash_balance_input', dtype=tf.float32)
    }

def process_shared_features(shared_features_input):
    import tensorflow as tf
    from tensorflow.keras import Input, Model
    from tensorflow.keras.layers import (
    Dense, LSTM, LayerNormalization, Concatenate, Flatten, Lambda, ConvLSTM2D
    )
    from tensorflow.keras.regularizers import l2
    from tensorflow.keras.utils import register_keras_serializable
    """
    Process shared features through LSTM layers.
    """
    x = LayerNormalization(dtype='float32', name='shared_norm_1')(shared_features_input)
    x = LSTM(D_HIDDEN_1, activation='tanh', return_sequences=True, name='shared_lstm_1', dtype='float32')(x)
    x = LayerNormalization(dtype='float32', name='shared_norm_2')(x)
    x = LSTM(D_HIDDEN_2, activation='tanh', name='shared_lstm_2', dtype='float32')(x)
    x = LayerNormalization(dtype='float32', name='shared_norm_3')(x)
    return Dense(64, activation='relu', kernel_regularizer=REG, name='shared_features_output')(x)

def process_ticker_features(ticker_features_input):
    import tensorflow as tf
    from tensorflow.keras import Input, Model
    from tensorflow.keras.layers import (
    Dense, LSTM, LayerNormalization, Concatenate, Flatten, Lambda, ConvLSTM2D
    )
    from tensorflow.keras.regularizers import l2
    from tensorflow.keras.utils import register_keras_serializable
    """Process ticker-specific features using ConvLSTM2D."""
    x = Lambda(expand_dims_channel, name='expand_dims')(ticker_features_input)
    x = ConvLSTM2D(filters=64, kernel_size=(1, 3), padding='same', return_sequences=True, activation='tanh', name='ticker_convlstm_1')(x)
    x = LayerNormalization(name='ticker_norm_1')(x)
    x = ConvLSTM2D(filters=32, kernel_size=(1, 3), padding='same', return_sequences=False, activation='tanh', name='ticker_convlstm_2')(x)
    x = LayerNormalization(name='ticker_norm_2')(x)
    x = Flatten(name='ticker_flatten')(x)
    return Dense(128, activation='relu', kernel_regularizer=REG, name='ticker_dense_out')(x)

def process_portfolio_state(portfolio_input, cash_input):
    import tensorflow as tf
    from tensorflow.keras import Input, Model
    from tensorflow.keras.layers import (
    Dense, LSTM, LayerNormalization, Concatenate, Flatten, Lambda, ConvLSTM2D
    )
    from tensorflow.keras.regularizers import l2
    from tensorflow.keras.utils import register_keras_serializable
    """Process portfolio holdings and cash balance."""
    h = Dense(64, activation='relu', kernel_regularizer=REG, name='portfolio_dense')(portfolio_input)
    h = LayerNormalization(name='portfolio_norm')(h)
    c = Dense(16, activation='relu', kernel_regularizer=REG, name='cash_dense')(cash_input)
    c = LayerNormalization(name='cash_norm')(c)
    combined_portfolio = Concatenate(name='portfolio_cash_concat')([h, c])
    return Dense(64, activation='relu', name='portfolio_cash_out')(combined_portfolio)

def build_common_feature_extractor(state_inputs,
                                   process_shared_fn=process_shared_features,
                                   process_ticker_fn=process_ticker_features,
                                   process_portfolio_fn=process_portfolio_state):
    import tensorflow as tf
    from tensorflow.keras import Input, Model
    from tensorflow.keras.layers import (
    Dense, LSTM, LayerNormalization, Concatenate, Flatten, Lambda, ConvLSTM2D
    )
    from tensorflow.keras.regularizers import l2
    from tensorflow.keras.utils import register_keras_serializable
    """Builds the common part of the network that processes state inputs."""
    shared_processed = process_shared_fn(state_inputs['shared_features'])
    ticker_processed = process_ticker_fn(state_inputs['ticker_features'])
    portfolio_processed = process_portfolio_fn(state_inputs['portfolio_holdings'], state_inputs['cash_balance'])

    combined = Concatenate(name='feature_combiner')([
        shared_processed,
        ticker_processed,
        portfolio_processed
    ])
    return combined

def build_actor(observation_space_dict, action_dim):
    import tensorflow as tf
    from tensorflow.keras import Input, Model
    from tensorflow.keras.layers import (
    Dense, LSTM, LayerNormalization, Concatenate, Flatten, Lambda, ConvLSTM2D
    )
    from tensorflow.keras.regularizers import l2
    from tensorflow.keras.utils import register_keras_serializable
    """
    Builds the actor network (policy).

    Args:
        observation_space_dict (dict): A dictionary where keys are feature names
                                     and values are gymnasium.spaces.Box instances.
        action_dim (int): The dimension of the action space.

    Returns:
        tf.keras.Model: The actor model.
    """
    window_size = observation_space_dict['shared_features'].shape[0]
    num_shared_features = observation_space_dict['shared_features'].shape[1]
    num_tradeable = observation_space_dict['ticker_features'].shape[1]
    num_ticker_features = observation_space_dict['ticker_features'].shape[2]

    state_inputs = get_input_layers(window_size, num_shared_features, num_tradeable, num_ticker_features)
    common_features = build_common_feature_extractor(state_inputs)

    x = Dense(D_HIDDEN_1, activation='relu', kernel_regularizer=REG, name='actor_dense_1')(common_features)
    x = LayerNormalization(name='actor_norm_1')(x)
    x = Dense(D_HIDDEN_2, activation='relu', kernel_regularizer=REG, name='actor_dense_2')(x)
    x = LayerNormalization(name='actor_norm_2')(x)

    means = Dense(action_dim, activation='linear', kernel_regularizer=REG, name='action_means')(x)
    log_stds = Dense(action_dim, activation='linear', kernel_regularizer=REG, name='action_log_stds')(x)

    model = Model(inputs=list(state_inputs.values()), outputs=[means, log_stds], name='actor')
    _LOG.debug("\nActor summary:")
    model.summary(line_length=120)
    return model

def build_critic(observation_space_dict, action_dim, name="critic"):
    import tensorflow as tf
    from tensorflow.keras import Input, Model
    from tensorflow.keras.layers import (
    Dense, LSTM, LayerNormalization, Concatenate, Flatten, Lambda, ConvLSTM2D
    )
    from tensorflow.keras.regularizers import l2
    from tensorflow.keras.utils import register_keras_serializable
    """
    Builds a critic network (Q-value function).

    Args:
        observation_space_dict (dict): A dictionary where keys are feature names
                                     and values are gymnasium.spaces.Box instances.
        action_dim (int): The dimension of the action space.
        name (str): Name for the critic network.

    Returns:
        tf.keras.Model: The critic model.
    """
    window_size = observation_space_dict['shared_features'].shape[0]
    num_shared_features = observation_space_dict['shared_features'].shape[1]
    num_tradeable = observation_space_dict['ticker_features'].shape[1]
    num_ticker_features = observation_space_dict['ticker_features'].shape[2]

    state_inputs = get_input_layers(window_size, num_shared_features, num_tradeable, num_ticker_features)
    action_input = Input(shape=(action_dim,), dtype=tf.float32, name='action_input')

    common_features = build_common_feature_extractor(state_inputs)

    action_features = Dense(64, activation='relu', kernel_regularizer=REG, name=f'{name}_action_dense')(action_input)
    action_features = LayerNormalization(name=f'{name}_action_norm')(action_features)

    combined_state_action = Concatenate(name=f'{name}_state_action_concat')([
        common_features,
        action_features
    ])

    x = Dense(D_HIDDEN_1, activation='relu', kernel_regularizer=REG, name=f'{name}_dense_1')(combined_state_action)
    x = LayerNormalization(name=f'{name}_norm_1')(x)
    x = Dense(D_HIDDEN_2, activation='relu', kernel_regularizer=REG, name=f'{name}_dense_2')(x)
    x = LayerNormalization(name=f'{name}_norm_2')(x)

    q_value = Dense(1, activation='linear', kernel_regularizer=REG, name=f'{name}_q_value')(x)

    model = Model(inputs=[*list(state_inputs.values()), action_input], outputs=q_value, name=name)
    _LOG.debug("\nActor summary:")
    model.summary(print_fn=_LOG.debug, line_length=120)
    return model
