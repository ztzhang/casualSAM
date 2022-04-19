# %%
import configparser
from os.path import join, abspath, dirname

# %%


def get_project_config(file_path=None):
    if file_path is None:
        file_path = join(dirname(abspath(__file__)), '../configs/project_config.cfg')
    config = configparser.ConfigParser()
    config.read(file_path)
    assert 'Paths' in config
    config_dict = {}
    for k, v in config['Paths'].items():
        config_dict[k] = v
    return config_dict


# should we set up config for training as well?


# %%
