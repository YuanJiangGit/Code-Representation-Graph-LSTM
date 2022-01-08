from configparser import ConfigParser


class MyConf:
    def __init__(self, config_file):
        config = ConfigParser()
        config.read(config_file, encoding='utf-8')
        self._config = config
        self.config_file=config_file

        # config.write(open(config_file,'w'))

    @property
    def wordEmbedding(self):
        return self._config.getboolean('data', 'word_embedding')

    @property
    def charData(self):
        return self._config.getboolean('data', 'char_data')

    @property
    def train_programs(self):
        return self._config.get('data', 'train_programs')

    @property
    def target_programs(self):
        return self._config.get('data', 'target_programs')

    @property
    def data_path(self):
        return self._config.get('data', 'data_path')

    @property
    def language(self):
        return self._config.get('data', 'language')

    @property
    def models_path(self):
        return self._config.get('data', 'models_path')

    @property
    def embedding_path(self):
        return self._config.get('data', 'embedding_path')

    @property
    def ratio(self):
        value = self._config.get("data", "ratio")
        # print(list(value))
        value = [int(k) for k in list(value) if k != ":"]
        return value

    @property
    def learning_rate(self):
        return self._config.getfloat('Optimizer', 'learning_rate')

    @property
    def weight_decay(self):
        return self._config.getfloat('Optimizer', 'weight_decay')

    @property
    def batch_size(self):
        return self._config.getint('params', 'batch_size')

    @property
    def class_num(self):
        return self._config.getint('params', 'class_num')

    @property
    def state_encode_bi(self):
        return self._config.getboolean('params', 'state_encode_bi')

    @property
    def hidden_dim(self):
        return self._config.getint('params', 'hidden_dim')

    @property
    def lstm_num_layers(self):
        return self._config.getint('params', 'lstm_num_layers')

    @property
    def embedding_ast_retrain(self):
        return self._config.getboolean('params', 'embedding_ast_retrain')

    @property
    def batch_normalizations(self):
        return self._config.getboolean('params', 'batch_normalizations')

    @property
    def bath_norm_momentum(self):
        return self._config.getfloat("params", "bath_norm_momentum")

    @property
    def batch_norm_affine(self):
        return self._config.getboolean("params", "batch_norm_affine")

    @property
    def dropout(self):
        return self._config.getfloat("params", "dropout")

    @property
    def dropout_embed(self):
        return self._config.getfloat("params", "dropout_embed")

    @property
    def kernel_num(self):
        return self._config.getint("params", "kernel_num")

    @property
    def kernel_sizes(self):
        value = self._config.getint("params", "kernel_sizes")
        return value

    @property
    def g_layers(self):
        value = self._config.getint("params", "g_layers")
        return value

    @property
    def k_neighbor(self):
        value = self._config.getint("params", "k_neighbor")
        return value

    @property
    def max_sen_length(self):
        return self._config.getint('params', 'max_sen_length')

    @property
    def init_weight(self):
        return self._config.getboolean("params", "init_weight")

    @property
    def init_weight_value(self):
        return self._config.getfloat("params", "init_weight_value")

    # Train
    @property
    def epochs(self):
        return self._config.getint("Train", "epochs")

    # use
    @property
    def use_plot(self):
        return self._config.getboolean('use', 'use_plot')

    @property
    def use_save(self):
        return self._config.getboolean('use', 'use_save')

    @property
    def use_gpu(self):
        return self._config.getboolean('use', 'use_gpu')

    def set_value(self, section, param, value):
        self._config.set(section, param, value)
        self._config.write(open(self.config_file, 'w'))

if __name__ == '__main__':
    conf = MyConf('config.cfg')
    print(conf.wordEmbedding)
    print(conf.charData)
