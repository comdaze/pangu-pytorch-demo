from .ordered_easydict import OrderedEasyDict as edict
import os
import torch


__C = edict()
cfg = __C
__C.GLOBAL = edict()
__C.GLOBAL.DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu")
__C.GLOBAL.BATCH_SZIE = 1
for dirs in ['/home/ec2-user/pangu-pytorch', '/opt/ml']:
    if os.path.exists(dirs):
        __C.GLOBAL.PATH = dirs
assert __C.GLOBAL.PATH is not None
__C.GLOBAL.SEED = 99
__C.GLOBAL.NUM_THREADS = 16


# __C.ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

if __C.GLOBAL.PATH.startswith('/opt/ml'):
    __C.PG_INPUT_PATH = os.path.join(__C.GLOBAL.PATH, 'input/data/')
else:
    __C.PG_INPUT_PATH = __C.GLOBAL.PATH
assert __C.PG_INPUT_PATH is not None

__C.PG_OUT_PATH = os.path.join(__C.GLOBAL.PATH, 'model')  # 'result'
assert __C.PG_OUT_PATH is not None

__C.ERA5_UPPER_LEVELS = ['1000', '925', '850', '700', '600',
                         '500', '400', '300', '250', '200', '150', '100', '50']
__C.ERA5_SURFACE_VARIABLES = ['msl', 'u10', 'v10', 't2m']
__C.ERA5_UPPER_VARIABLES = ['z', 'q', 't', 'u', 'v']


__C.PG = edict()

__C.PG.TRAIN = edict()

# 时间步长（例如，每步预测 24 小时）, 可选：1, 3, 6, 24。这里名称定义可能会有歧义，按理说HORIZON是指预测范围，也就是几个“时间步长”，但是本项目中就当“时间步长”来用了
__C.PG.HORIZON = 24

__C.PG.TRAIN.EPOCHS = 2  # default: 100
__C.PG.TRAIN.LR = 5e-6  # 5e-4
__C.PG.TRAIN.WEIGHT_DECAY = 3e-6
__C.PG.TRAIN.START_TIME = '20180101'
__C.PG.TRAIN.END_TIME = '20230101'
__C.PG.TRAIN.FREQUENCY = '12h'  # default: 12h (HORIZON=24)
__C.PG.TRAIN.BATCH_SIZE = 16  # match GPU num
__C.PG.TRAIN.UPPER_WEIGHTS = [3.00, 0.60, 1.50, 0.77, 0.54]
__C.PG.TRAIN.SURFACE_WEIGHTS = [1.50, 0.77, 0.66, 3.00]
__C.PG.TRAIN.SAVE_INTERVAL = 1
__C.PG.VAL = edict()


__C.PG.VAL.START_TIME = '20230101'
__C.PG.VAL.END_TIME = '20240101'
__C.PG.VAL.FREQUENCY = '12h'  # default: 12h (HORIZON=24)
__C.PG.VAL.BATCH_SIZE = 1
__C.PG.VAL.INTERVAL = 1


__C.PG.TEST = edict()

__C.PG.TEST.START_TIME = '20240101'
__C.PG.TEST.END_TIME = '20240601'
__C.PG.TEST.FREQUENCY = '12h'  # default: 12h (HORIZON=24)
__C.PG.TEST.BATCH_SIZE = 1

__C.PG.BENCHMARK = edict()

__C.PG.BENCHMARK.PRETRAIN_24 = os.path.join(
    __C.PG_INPUT_PATH, 'pretrained_model/pangu_weather_24.onnx')
__C.PG.BENCHMARK.PRETRAIN_6 = os.path.join(
    __C.PG_INPUT_PATH, 'pretrained_model/pangu_weather_6.onnx')
__C.PG.BENCHMARK.PRETRAIN_3 = os.path.join(
    __C.PG_INPUT_PATH, 'pretrained_model/pangu_weather_3.onnx')
__C.PG.BENCHMARK.PRETRAIN_1 = os.path.join(
    __C.PG_INPUT_PATH, 'pretrained_model/pangu_weather_1.onnx')

__C.PG.BENCHMARK.PRETRAIN_24_fp16 = os.path.join(
    __C.PG_INPUT_PATH, 'pretrained_model_fp16/pangu_weather_24_fp16.onnx')

__C.PG.BENCHMARK.PRETRAIN_24_torch = os.path.join(
    __C.PG_INPUT_PATH, 'pretrained_model/pangu_weather_24_torch.pth')
__C.PG.BENCHMARK.PRETRAIN_6_torch = os.path.join(
    __C.PG_INPUT_PATH, 'pretrained_model/pangu_weather_6_torch.pth')
__C.PG.BENCHMARK.PRETRAIN_3_torch = os.path.join(
    __C.PG_INPUT_PATH, 'pretrained_model/pangu_weather_3_torch.pth')
__C.PG.BENCHMARK.PRETRAIN_1_torch = os.path.join(
    __C.PG_INPUT_PATH, 'pretrained_model/pangu_weather_1_torch.pth')


__C.MODEL = edict()
