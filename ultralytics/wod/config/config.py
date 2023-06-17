from easydict import EasyDict as edict


def load():
    config = edict()

    config.range_x = [0, 50]
    config.range_y = [-25, 25]
    config.range_z = [-1, 3]
    config.bev_width = 640
    config.bev_height = 640

    return config
