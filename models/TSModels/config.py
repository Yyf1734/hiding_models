root_dir = "/home/myang_20210409/yyf/model_overloading"
default_data_dir = None
class Config:
    def __init__(self,
                 hidden_size=20,
                 lr=0.001,
                 epoch=100,
                 input_size = 1,
                 class_num = 3,
                 encoder_type='rnn',
                 predictor_type='rnn'):
        self.hidden_size = hidden_size
        self.lr = lr
        self.epoch = epoch
        self.input_size = input_size
        self.class_num = class_num


class MLPConfig:
    def __init__(self,
                 hidden_size=100,                 
                 activate_type=None):
        self.hidden_size = hidden_size
        self.activate_type = activate_type
        


class RNNConfig:
    def __init__(self,
                 rnn_layer=1,
                 rnn_type='gru'):
        self.rnn_layer = rnn_layer
        self.rnn_type = rnn_type


class TCNConfig:
    def __init__(self,
                 hidden_sizes=[20, 20, 20],
                 dropout=0.1,
                 kernel_size=2,
                 stride=1):
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        self.kernel_size = kernel_size
        self.stride = stride


class TFMConfig:
    def __init__(self,
                 dropout=0.1,
                 nhead=2,
                 dim_ff=50,
                 activation='relu',
                 tfm_layer=3):
        self.dropout = dropout
        self.nhead = nhead
        self.dim_ff = dim_ff
        self.activation = activation
        self.tfm_layer = tfm_layer


class CNNConfig:
    def __init__(self,
                 strides=[3, 3],
                 kernel_sizes=[9, 5],
                 channels=1,
                 activation='relu'):
        self.strides = strides
        self.kernel_sizes = kernel_sizes
        self.channels = channels
        self.activation = activation


class GAConfig:
    def __init__(self,
                 popsize=400,
                 attack_num=3,
                 attack_value_min=-2,
                 attack_value_max=2,
                 attack_pos_min=0,
                 attack_pos_max=0,
                 attack_pos_init_min=0,
                 attack_pos_init_max=0,
                 maxiter=100):
        self.popsize = popsize
        self.maxiter = maxiter
        self.attack_num = attack_num
        self.attack_value_min = attack_value_min
        self.attack_value_max = attack_value_max
        self.attack_pos_min = attack_pos_min
        self.attack_pos_max = attack_pos_max
        self.attack_pos_init_min = attack_pos_init_min
        self.attack_pos_init_max = attack_pos_init_max


class FixedConfig:
    def __init__(self,
                 attack_len=0,
                 attack_mean=0,
                 attack_std=0):
        self.attack_len = attack_len
        self.attack_mean = attack_mean
        self.attack_std = attack_std


class FGSMConfig:
    def __int__(self,
                attack_len=0,
                attack_epsilon=0,
                attack_value_min=0,
                attack_value_max=0,
                attack_masked=True):
        self.attack_len = attack_len
        self.attack_epsilon = attack_epsilon
        self.attack_value_min = attack_value_min
        self.attack_value_max = attack_value_max
        self.attack_masked = attack_masked


class JSMAConfig:
    def __int__(self,
                attack_len=0,
                attack_epsilon=0):
        self.attack_len = attack_len
        self.attack_epsilon = attack_epsilon


class CWConfig:
    def __init__(self,
                 norm=0,
                 confidence=0,
                 threshold=0,
                 learning_rate=0,
                 epochs=0,
                 k=0,
                 epsilon=0):
        self.norm = norm
        self.confidence = confidence
        self.threshold = threshold
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.k = k
        self.epsilon = epsilon

def gen_mlp_config(gpu_id=0):
    config = Config()

    config.encoder_type = 'mlp'
    config.predictor_type = 'mlp'
    config.data_dir = default_data_dir
    config.batch_size = 64
    config.lr = 0.002
    config.weight_decay = 0.99
    config.gpu_id = gpu_id
    config.epoch = 100
    config.opt = 'adam'

    config.max_len = 200

    mlp_config = MLPConfig()
    mlp_config.activation_type = 'tanh'
    mlp_config.hidden_size = 80

    config.mlp_config = mlp_config

    return config


def gen_cnn_config(gpu_id=0):
    config = Config()

    config.encoder_type = 'cnn'
    config.predictor_type = 'mlp'
    config.data_dir = default_data_dir
    config.batch_size = 64
    config.lr = 0.02
    config.weight_decay = 0.99
    config.gpu_id = gpu_id
    config.epoch = 100
    config.opt = 'adam'

    config.max_len = 200

    cnn_config = CNNConfig()
    cnn_config.strides = [3, 3]
    cnn_config.kernel_sizes = [9, 5]

    config.cnn_config = cnn_config

    return config


def gen_rnn_config(gpu_id=0):
    config = Config()

    config.encoder_type = 'rnn'
    config.predictor_type = 'rnn'
    config.data_dir = default_data_dir
    config.batch_size = 64
    config.lr = 0.002
    config.weight_decay = 0.005
    config.gpu_id = gpu_id
    config.epoch = 100
    config.opt = 'rmsprop'

    config.max_len = 200

    rnn_config = RNNConfig()
    rnn_config.rnn_type = 'rnn'

    config.rnn_config = rnn_config

    return config


def gen_gru_config(gpu_id=0):
    config = Config()

    config.encoder_type = 'rnn'
    config.predictor_type = 'rnn'
    config.data_dir = default_data_dir
    config.batch_size = 64
    config.lr = 0.002
    config.weight_decay = 0.99
    config.gpu_id = gpu_id
    config.epoch = 100
    config.opt = 'rmsprop'

    config.max_len = 200

    rnn_config = RNNConfig()
    rnn_config.rnn_type = 'gru'

    config.rnn_config = rnn_config

    return config


def gen_lstm_config(gpu_id=0):
    config = Config()

    config.encoder_type = 'rnn'
    config.predictor_type = 'rnn'
    config.data_dir = default_data_dir
    config.batch_size = 64
    config.lr = 0.002
    config.weight_decay = 0.99
    config.gpu_id = gpu_id
    config.epoch = 200
    config.opt = 'rmsprop'

    config.max_len = 200

    rnn_config = RNNConfig()
    rnn_config.rnn_type = 'lstm'

    config.rnn_config = rnn_config

    return config


def gen_tcn_config(gpu_id=0):
    config = Config()

    config.encoder_type = 'tcn'
    config.predictor_type = 'attention'
    config.data_dir = default_data_dir
    config.batch_size = 256
    config.lr = 0.002
    config.weight_decay = 0.01
    config.gpu_id = gpu_id
    config.epoch = 80
    config.opt = 'rmsprop'

    config.max_len = 200

    tcn_config = TCNConfig()
    tcn_config.kernel_size = 5
    tcn_config.dropout = 0.1
    tcn_config.hidden_sizes = [20]

    config.tcn_config = tcn_config

    return config

configs = {
    'mlp': gen_mlp_config,
    'cnn': gen_cnn_config,
    'rnn': gen_rnn_config,
    'gru': gen_gru_config,
    'lstm': gen_lstm_config,
    'tcn': gen_tcn_config,
}

# which defines the input_dim and output_dim for each dataset
dataset_settings = {
    'fake': {
        'in': 1,
        'out': 3,
    },
    'stock': {
        'in': 1,
        'out': 3,
    },
    'stock_small': {
        'in': 1,
        'out': 3,
    },
    'stock_4dim': {
        'in': 4,
        'out': 3,
    },
    'traffic': {
        'in': 1,
        'out': 3,
    },
    'traffic_2dim': {
        'in': 2,
        'out': 3,
    },
    'climate': {
        'in': 1,
        'out': 3,
    },
    'climate_100': {
        'in': 1,
        'out': 3,
    },
    'climate_200': {
        'in': 1,
        'out': 3,
    },
    'eye_13dim': {
        'in': 13,
        'out': 2,
    },
    'eye_14dim': {
        'in': 14,
        'out': 2,
    },
    'ECG1': {
        'in': 1,
        'out': 42,
    },
    'Epileps_3dim': {
        'in': 3,
        'out': 4,
    },
    'day': {
        'in': 1,
        'out': 7,
    },
    'day_2class': {
        'in': 1,
        'out': 2,
    },
    'fungi': {
        'in': 1,
        'out': 18,
    },
    'power': {
        'in': 1,
        'out': 2,
    },
    'trace': {
        'in': 1,
        'out': 4,
    },
    'ham': {
        'in': 1,
        'out': 2,
    },
    'ECGFiveDays': {
        'in': 1,
        'out': 2,
    },
    'SonyAIBORobotSurface1': {
        'in': 1,
        'out': 2,
    },
    'RacketSports_6dim': {
        'in': 6,
        'out': 4,
    },
    'UWaveGestureLibrary_3dim': {
        'in': 3,
        'out': 8,
    },
    'Epilepsy_3dim': {
        'in': 3,
        'out': 4,
    },
    'Libras_2dim': {
        'in': 2,
        'out': 15,
    },
    'BasicMotions_6dim': {
        'in': 6,
        'out': 4,
    },
}


def get_dataset_config(dataset_name, config):
    config.data_dir = '%s/data/data_%s.pkl' % (root_dir, dataset_name)
    config.input_size = dataset_settings[dataset_name]['in']
    config.class_num = dataset_settings[dataset_name]['out']

    if dataset_name == 'ECG1':
        config.max_len = 750
        config.epoch *= 4
    if dataset_name.startswith('Epileps'):
        config.max_len = 206
        # config.epoch = int(config.epoch * 1.5)
        if hasattr(config, 'cnn_config'):
            config.cnn_config.kernel_sizes = [5, 5, 5]
            config.cnn_config.strides = [4, 2, 1]
            config.cnn_config.channels = [3, 2, 1]
    if dataset_name.startswith('day'):
        config.max_len = 288
        config.epoch = int(config.epoch * 2)
    if dataset_name == 'fungi':
        config.max_len = 201
    if dataset_name == 'power':
        config.max_len = 144
    if dataset_name == 'trace':
        config.max_len = 275
    if dataset_name == 'ham':
        config.max_len = 431
    if dataset_name == 'ECGFiveDays':
        config.max_len = 136
    if dataset_name == 'SonyAIBORobotSurface1':
        config.max_len = 70
    if dataset_name == 'RacketSports_6dim':
        config.max_len = 30
        if hasattr(config, 'cnn_config'):
            config.cnn_config.kernel_sizes = [7, 5]
            config.cnn_config.strides = [1, 1]
            config.cnn_config.channels = [1, 1]
        if hasattr(config, 'ga_config'):
            config.ga_config.attack_pos_init_min = config.max_len - 30
            config.ga_config.attack_pos_init_max = config.max_len - 1
    if dataset_name == 'UWaveGestureLibrary_3dim':
        config.max_len = 315
        if hasattr(config, 'cnn_config'):
            config.cnn_config.kernel_sizes = [5, 5, 5, 4]
            config.cnn_config.strides = [3, 2, 2, 1]
            config.cnn_config.channels = [3, 3, 2, 1]
        if hasattr(config, 'tcn_config'):
            config.tcn_config.kernel_size = 18
            config.tcn_config.hidden_sizes = [20, 20]
            config.tcn_config.dropout = 0.2
            config.lr = 0.002
            config.weight_decay = 0.005
            config.epoch = 80
    if dataset_name == 'Libras_2dim':
        config.max_len = 45
    if dataset_name == 'BasicMotions_6dim':
        config.max_len = 100
        if hasattr(config, 'cnn_config'):
            config.cnn_config.kernel_sizes = [7, 5, 3]
            config.cnn_config.strides = [2, 2, 1]
            config.cnn_config.channels = [4, 2, 1]

    return config


def get_config(model_name, gpu_id=0):
    return configs[model_name](gpu_id)

