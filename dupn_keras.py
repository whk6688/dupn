import os
import time
import numpy as np
import pandas as pd
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Reshape, Lambda, concatenate, dot, add, \
    Multiply
from keras.layers import Dropout, GaussianDropout, multiply, SpatialDropout1D, BatchNormalization
from keras.models import Model
import numpy as np
import tensorflow as tf
from keras.optimizers import Adam
from sklearn.utils import shuffle
import datetime
from sklearn.metrics import roc_auc_score
from tensorflow.python.keras.models import save_model, load_model
from keras import regularizers
from keras.utils import plot_model


class EarlyStoppingAtMinLoss(tf.keras.callbacks.Callback):
    """Stop training when the loss is at its min, i.e. the loss stops decreasing.

  Arguments:
      patience: Number of epochs to wait after min has been hit. After this
      number of no improvement, training stops.
  """

    def __init__(self, patience=0):
        super(EarlyStoppingAtMinLoss, self).__init__()
        self.patience = patience
        # best_weights to store the weights at which the minimum loss occurs.
        self.best_weights = None

    def on_train_begin(self, logs=None):
        # The number of epoch it has waited when loss is no longer minimum.
        self.wait = 0
        # The epoch the training stops at.
        self.stopped_epoch = 0
        # Initialize the best as infinity.
        self.best = 0

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get("val_auc")
        if np.greater(current, self.best):
            self.best = current
            self.wait = 0
            # Record the best weights if current results is better (less).
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                print("Restoring model weights from the end of the best epoch.")
                self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print("Epoch %05d: early stopping" % (self.stopped_epoch + 1))


def auc(y_true, y_pred):
    return tf.py_func(roc_auc_score, (y_true, y_pred), tf.double)


os.environ['CUDA_VISIBLE_DEVICES'] = '1'
date_type = {'airQuality': np.int16, 'area': np.int16, 'bedroomType': np.int16, 'buildYear': np.int16,
             'compartmentFace': np.int16, 'disposeBedroomAmount': np.int16, 'floor': np.int16, 'haveLift': np.int16,
             'houseType': np.int16, 'isActivity': np.int16, 'isAiLock': np.int16, 'isBalcony': np.int16,
             'isCz': np.int16, 'isDw': np.int16, 'isHighRate': np.int16, 'isNew': np.int16, 'isSubsidy': np.int16,
             'priceRange': np.int16, 'resblockName': np.int16, 'styleCode': np.int16, 'supplyHeat': np.int16,
             'districtCode': np.int16, 'cityCode': np.int16, 'isTop': np.int16, 'isButtom': np.int16,
             'canPet': np.int16, 'isDeepBreath': np.int16, 'isDiscount': np.int16,
             'ta3_airQuality': np.int16, 'ta3_area': np.int16, 'ta3_bedroomType': np.int16, 'ta3_buildYear': np.int16,
             'ta3_compartmentFace': np.int16, 'ta3_disposeBedroomAmount': np.int16, 'ta3_floor': np.int16,
             'ta3_haveLift': np.int16, 'ta3_houseType': np.int16, 'ta3_isActivity': np.int16, 'ta3_isAiLock': np.int16,
             'ta3_isBalcony': np.int16, 'ta3_isCz': np.int16, 'ta3_isDw': np.int16, 'ta3_isHighRate': np.int16,
             'ta3_isNew': np.int16, 'ta3_isSubsidy': np.int16, 'ta3_priceRange': np.int16, 'ta3_styleCode': np.int16,
             'ta3_supplyHeat': np.int16, 'ta3_cityCode': np.int16, 'ta3_isTop': np.int16, 'ta3_isButtom': np.int16,
             'ta3_canPet': np.int16, 'ta3_isDeepBreath': np.int16, 'ta3_isDiscount': np.int16,
             'ta7_airQuality': np.int16, 'ta7_area': np.int16, 'ta7_bedroomType': np.int16, 'ta7_buildYear': np.int16,
             'ta7_compartmentFace': np.int16, 'ta7_disposeBedroomAmount': np.int16, 'ta7_floor': np.int16,
             'ta7_haveLift': np.int16, 'ta7_houseType': np.int16, 'ta7_isActivity': np.int16, 'ta7_isAiLock': np.int16,
             'ta7_isBalcony': np.int16, 'ta7_isCz': np.int16, 'ta7_isDw': np.int16, 'ta7_isHighRate': np.int16,
             'ta7_isNew': np.int16, 'ta7_isSubsidy': np.int16, 'ta7_priceRange': np.int16, 'ta7_styleCode': np.int16,
             'ta7_supplyHeat': np.int16, 'ta7_cityCode': np.int16, 'ta7_isTop': np.int16, 'ta7_isButtom': np.int16,
             'ta7_canPet': np.int16, 'ta7_isDeepBreath': np.int16, 'ta7_isDiscount': np.int16,
             'ta14_airQuality': np.int16, 'ta14_area': np.int16, 'ta14_bedroomType': np.int16,
             'ta14_buildYear': np.int16, 'ta14_compartmentFace': np.int16, 'ta14_disposeBedroomAmount': np.int16,
             'ta14_floor': np.int16, 'ta14_haveLift': np.int16, 'ta14_houseType': np.int16, 'ta14_isActivity': np.int16,
             'ta14_isAiLock': np.int16, 'ta14_isBalcony': np.int16, 'ta14_isCz': np.int16, 'ta14_isDw': np.int16,
             'ta14_isHighRate': np.int16, 'ta14_isNew': np.int16, 'ta14_isSubsidy': np.int16,
             'ta14_priceRange': np.int16, 'ta14_styleCode': np.int16, 'ta14_supplyHeat': np.int16,
             'ta14_cityCode': np.int16, 'ta14_isTop': np.int16, 'ta14_isButtom': np.int16, 'ta14_canPet': np.int16,
             'ta14_isDeepBreath': np.int16, 'ta14_isDiscount': np.int16,

             'tb3_airQuality': np.int16, 'tb3_area': np.int16, 'tb3_bedroomType': np.int16, 'tb3_buildYear': np.int16,
             'tb3_compartmentFace': np.int16, 'tb3_disposeBedroomAmount': np.int16, 'tb3_floor': np.int16,
             'tb3_haveLift': np.int16, 'tb3_houseType': np.int16, 'tb3_isActivity': np.int16, 'tb3_isAiLock': np.int16,
             'tb3_isBalcony': np.int16, 'tb3_isCz': np.int16, 'tb3_isDw': np.int16, 'tb3_isHighRate': np.int16,
             'tb3_isNew': np.int16, 'tb3_isSubsidy': np.int16, 'tb3_priceRange': np.int16, 'tb3_styleCode': np.int16,
             'tb3_supplyHeat': np.int16, 'tb3_cityCode': np.int16, 'tb3_isTop': np.int16, 'tb3_isButtom': np.int16,
             'tb3_canPet': np.int16, 'tb3_isDeepBreath': np.int16, 'tb3_isDiscount': np.int16,
             'tb7_airQuality': np.int16, 'tb7_area': np.int16, 'tb7_bedroomType': np.int16, 'tb7_buildYear': np.int16,
             'tb7_compartmentFace': np.int16, 'tb7_disposeBedroomAmount': np.int16, 'tb7_floor': np.int16,
             'tb7_haveLift': np.int16, 'tb7_houseType': np.int16, 'tb7_isActivity': np.int16, 'tb7_isAiLock': np.int16,
             'tb7_isBalcony': np.int16, 'tb7_isCz': np.int16, 'tb7_isDw': np.int16, 'tb7_isHighRate': np.int16,
             'tb7_isNew': np.int16, 'tb7_isSubsidy': np.int16, 'tb7_priceRange': np.int16, 'tb7_styleCode': np.int16,
             'tb7_supplyHeat': np.int16, 'tb7_cityCode': np.int16, 'tb7_isTop': np.int16, 'tb7_isButtom': np.int16,
             'tb7_canPet': np.int16, 'tb7_isDeepBreath': np.int16, 'tb7_isDiscount': np.int16,
             'tb14_airQuality': np.int16, 'tb14_area': np.int16, 'tb14_bedroomType': np.int16,
             'tb14_buildYear': np.int16, 'tb14_compartmentFace': np.int16, 'tb14_disposeBedroomAmount': np.int16,
             'tb14_floor': np.int16, 'tb14_haveLift': np.int16, 'tb14_houseType': np.int16, 'tb14_isActivity': np.int16,
             'tb14_isAiLock': np.int16, 'tb14_isBalcony': np.int16, 'tb14_isCz': np.int16, 'tb14_isDw': np.int16,
             'tb14_isHighRate': np.int16, 'tb14_isNew': np.int16, 'tb14_isSubsidy': np.int16,
             'tb14_priceRange': np.int16, 'tb14_styleCode': np.int16, 'tb14_supplyHeat': np.int16,
             'tb14_cityCode': np.int16, 'tb14_isTop': np.int16, 'tb14_isButtom': np.int16, 'tb14_canPet': np.int16,
             'tb14_isDeepBreath': np.int16, 'tb14_isDiscount': np.int16,

             'tc3_airQuality': np.int16, 'tc3_area': np.int16, 'tc3_bedroomType': np.int16, 'tc3_buildYear': np.int16,
             'tc3_compartmentFace': np.int16, 'tc3_disposeBedroomAmount': np.int16, 'tc3_floor': np.int16,
             'tc3_haveLift': np.int16, 'tc3_houseType': np.int16, 'tc3_isActivity': np.int16, 'tc3_isAiLock': np.int16,
             'tc3_isBalcony': np.int16, 'tc3_isCz': np.int16, 'tc3_isDw': np.int16, 'tc3_isHighRate': np.int16,
             'tc3_isNew': np.int16, 'tc3_isSubsidy': np.int16, 'tc3_priceRange': np.int16, 'tc3_styleCode': np.int16,
             'tc3_supplyHeat': np.int16, 'tc3_cityCode': np.int16, 'tc3_isTop': np.int16, 'tc3_isButtom': np.int16,
             'tc3_canPet': np.int16, 'tc3_isDeepBreath': np.int16, 'tc3_isDiscount': np.int16,
             'tc7_airQuality': np.int16, 'tc7_area': np.int16, 'tc7_bedroomType': np.int16, 'tc7_buildYear': np.int16,
             'tc7_compartmentFace': np.int16, 'tc7_disposeBedroomAmount': np.int16, 'tc7_floor': np.int16,
             'tc7_haveLift': np.int16, 'tc7_houseType': np.int16, 'tc7_isActivity': np.int16, 'tc7_isAiLock': np.int16,
             'tc7_isBalcony': np.int16, 'tc7_isCz': np.int16, 'tc7_isDw': np.int16, 'tc7_isHighRate': np.int16,
             'tc7_isNew': np.int16, 'tc7_isSubsidy': np.int16, 'tc7_priceRange': np.int16, 'tc7_styleCode': np.int16,
             'tc7_supplyHeat': np.int16, 'tc7_cityCode': np.int16, 'tc7_isTop': np.int16, 'tc7_isButtom': np.int16,
             'tc7_canPet': np.int16, 'tc7_isDeepBreath': np.int16, 'tc7_isDiscount': np.int16,
             'tc14_airQuality': np.int16, 'tc14_area': np.int16, 'tc14_bedroomType': np.int16,
             'tc14_buildYear': np.int16, 'tc14_compartmentFace': np.int16, 'tc14_disposeBedroomAmount': np.int16,
             'tc14_floor': np.int16, 'tc14_haveLift': np.int16, 'tc14_houseType': np.int16, 'tc14_isActivity': np.int16,
             'tc14_isAiLock': np.int16, 'tc14_isBalcony': np.int16, 'tc14_isCz': np.int16, 'tc14_isDw': np.int16,
             'tc14_isHighRate': np.int16, 'tc14_isNew': np.int16, 'tc14_isSubsidy': np.int16,
             'tc14_priceRange': np.int16, 'tc14_styleCode': np.int16, 'tc14_supplyHeat': np.int16,
             'tc14_cityCode': np.int16, 'tc14_isTop': np.int16, 'tc14_isButtom': np.int16, 'tc14_canPet': np.int16,
             'tc14_isDeepBreath': np.int16, 'tc14_isDiscount': np.int16,
             'mulSubwayStation': np.int16, 'mulSubwayLine': np.int16, 'mulBizcirclecode': np.int16,
             'ta3_mulResblockName': np.int16, 'ta3_mulDistrictCode': np.int16, 'ta3_mulBizcirclecode': np.int16,
             'ta3_mulSubwayStation': np.int16, 'ta3_mulSubwayLine': np.int16,
             'ta7_mulResblockName': np.int16, 'ta7_mulDistrictCode': np.int16, 'ta7_mulBizcirclecode': np.int16,
             'ta7_mulSubwayStation': np.int16, 'ta7_mulSubwayLine': np.int16,
             'ta14_mulResblockName': np.int16, 'ta14_mulDistrictCode': np.int16, 'ta14_mulBizcirclecode': np.int16,
             'ta14_mulSubwayStation': np.int16, 'ta14_mulSubwayLine': np.int16,

             'tb3_mulResblockName': np.int16, 'tb3_mulDistrictCode': np.int16, 'tb3_mulBizcirclecode': np.int16,
             'tb3_mulSubwayStation': np.int16, 'tb3_mulSubwayLine': np.int16,
             'tb7_mulResblockName': np.int16, 'tb7_mulDistrictCode': np.int16, 'tb7_mulBizcirclecode': np.int16,
             'tb7_mulSubwayStation': np.int16, 'tb7_mulSubwayLine': np.int16,
             'tb14_mulResblockName': np.int16, 'tb14_mulDistrictCode': np.int16, 'tb14_mulBizcirclecode': np.int16,
             'tb14_mulSubwayStation': np.int16, 'tb14_mulSubwayLine': np.int16,

             'tc3_mulResblockName': np.int16, 'tc3_mulDistrictCode': np.int16, 'tc3_mulBizcirclecode': np.int16,
             'tc3_mulSubwayStation': np.int16, 'tc3_mulSubwayLine': np.int16,
             'tc7_mulResblockName': np.int16, 'tc7_mulDistrictCode': np.int16, 'tc7_mulBizcirclecode': np.int16,
             'tc7_mulSubwayStation': np.int16, 'tc7_mulSubwayLine': np.int16,
             'tc14_mulResblockName': np.int16, 'tc14_mulDistrictCode': np.int16, 'tc14_mulBizcirclecode': np.int16,
             'tc14_mulSubwayStation': np.int16, 'tc14_mulSubwayLine': np.int16,
             'res': np.int16, 'roomPrice': np.float32, 'areaNumber': np.float32,
             'house_ctr': np.float32, 'user_ctr': np.float32,
             'hid': np.int32, 'invnolist': str, 'pricelist': str, 'resblocklist': str, 'stationlist': str,
             'linelist': str
             }

df_chunk = pd.read_csv("/data/home/wanghk/datadin/din.csv", sep=';', dtype=date_type, chunksize=100000)

res_chunk = []
for chunk in df_chunk:
    res_chunk.append(chunk)
train_df = pd.concat(res_chunk)
train_size = len(train_df)
print("load data end", train_size, datetime.datetime.now())
train_df = shuffle(train_df)
print(train_df)
sparse_features = ['airQuality', 'area', 'bedroomType', 'buildYear', 'compartmentFace', 'disposeBedroomAmount', 'floor',
                   'haveLift', 'houseType', 'isActivity', 'isAiLock', 'isBalcony', 'isCz', 'isDw', 'isHighRate',
                   'isNew', 'isSubsidy', 'priceRange', 'resblockName', 'styleCode', 'supplyHeat', 'districtCode',
                   'cityCode', 'isTop', 'isButtom', 'canPet', 'isDeepBreath', 'isDiscount',
                   'ta3_airQuality', 'ta3_area', 'ta3_bedroomType', 'ta3_buildYear', 'ta3_compartmentFace',
                   'ta3_disposeBedroomAmount', 'ta3_floor', 'ta3_haveLift', 'ta3_houseType', 'ta3_isActivity',
                   'ta3_isAiLock', 'ta3_isBalcony', 'ta3_isCz', 'ta3_isDw', 'ta3_isHighRate', 'ta3_isNew',
                   'ta3_isSubsidy', 'ta3_priceRange', 'ta3_styleCode', 'ta3_supplyHeat', 'ta3_cityCode', 'ta3_isTop',
                   'ta3_isButtom', 'ta3_canPet', 'ta3_isDeepBreath', 'ta3_isDiscount',
                   'ta7_airQuality', 'ta7_area', 'ta7_bedroomType', 'ta7_buildYear', 'ta7_compartmentFace',
                   'ta7_disposeBedroomAmount', 'ta7_floor', 'ta7_haveLift', 'ta7_houseType', 'ta7_isActivity',
                   'ta7_isAiLock', 'ta7_isBalcony', 'ta7_isCz', 'ta7_isDw', 'ta7_isHighRate', 'ta7_isNew',
                   'ta7_isSubsidy', 'ta7_priceRange', 'ta7_styleCode', 'ta7_supplyHeat', 'ta7_cityCode', 'ta7_isTop',
                   'ta7_isButtom', 'ta7_canPet', 'ta7_isDeepBreath', 'ta7_isDiscount',
                   'ta14_airQuality', 'ta14_area', 'ta14_bedroomType', 'ta14_buildYear', 'ta14_compartmentFace',
                   'ta14_disposeBedroomAmount', 'ta14_floor', 'ta14_haveLift', 'ta14_houseType', 'ta14_isActivity',
                   'ta14_isAiLock', 'ta14_isBalcony', 'ta14_isCz', 'ta14_isDw', 'ta14_isHighRate', 'ta14_isNew',
                   'ta14_isSubsidy', 'ta14_priceRange', 'ta14_styleCode', 'ta14_supplyHeat', 'ta14_cityCode',
                   'ta14_isTop', 'ta14_isButtom', 'ta14_canPet', 'ta14_isDeepBreath', 'ta14_isDiscount',

                   'tb3_airQuality', 'tb3_area', 'tb3_bedroomType', 'tb3_buildYear', 'tb3_compartmentFace',
                   'tb3_disposeBedroomAmount', 'tb3_floor', 'tb3_haveLift', 'tb3_houseType', 'tb3_isActivity',
                   'tb3_isAiLock', 'tb3_isBalcony', 'tb3_isCz', 'tb3_isDw', 'tb3_isHighRate', 'tb3_isNew',
                   'tb3_isSubsidy', 'tb3_priceRange', 'tb3_styleCode', 'tb3_supplyHeat', 'tb3_cityCode', 'tb3_isTop',
                   'tb3_isButtom', 'tb3_canPet', 'tb3_isDeepBreath', 'tb3_isDiscount',
                   'tb7_airQuality', 'tb7_area', 'tb7_bedroomType', 'tb7_buildYear', 'tb7_compartmentFace',
                   'tb7_disposeBedroomAmount', 'tb7_floor', 'tb7_haveLift', 'tb7_houseType', 'tb7_isActivity',
                   'tb7_isAiLock', 'tb7_isBalcony', 'tb7_isCz', 'tb7_isDw', 'tb7_isHighRate', 'tb7_isNew',
                   'tb7_isSubsidy', 'tb7_priceRange', 'tb7_styleCode', 'tb7_supplyHeat', 'tb7_cityCode', 'tb7_isTop',
                   'tb7_isButtom', 'tb7_canPet', 'tb7_isDeepBreath', 'tb7_isDiscount',
                   'tb14_airQuality', 'tb14_area', 'tb14_bedroomType', 'tb14_buildYear', 'tb14_compartmentFace',
                   'tb14_disposeBedroomAmount', 'tb14_floor', 'tb14_haveLift', 'tb14_houseType', 'tb14_isActivity',
                   'tb14_isAiLock', 'tb14_isBalcony', 'tb14_isCz', 'tb14_isDw', 'tb14_isHighRate', 'tb14_isNew',
                   'tb14_isSubsidy', 'tb14_priceRange', 'tb14_styleCode', 'tb14_supplyHeat', 'tb14_cityCode',
                   'tb14_isTop', 'tb14_isButtom', 'tb14_canPet', 'tb14_isDeepBreath', 'tb14_isDiscount',

                   'tc3_airQuality', 'tc3_area', 'tc3_bedroomType', 'tc3_buildYear', 'tc3_compartmentFace',
                   'tc3_disposeBedroomAmount', 'tc3_floor', 'tc3_haveLift', 'tc3_houseType', 'tc3_isActivity',
                   'tc3_isAiLock', 'tc3_isBalcony', 'tc3_isCz', 'tc3_isDw', 'tc3_isHighRate', 'tc3_isNew',
                   'tc3_isSubsidy', 'tc3_priceRange', 'tc3_styleCode', 'tc3_supplyHeat', 'tc3_cityCode', 'tc3_isTop',
                   'tc3_isButtom', 'tc3_canPet', 'tc3_isDeepBreath', 'tc3_isDiscount',
                   'tc7_airQuality', 'tc7_area', 'tc7_bedroomType', 'tc7_buildYear', 'tc7_compartmentFace',
                   'tc7_disposeBedroomAmount', 'tc7_floor', 'tc7_haveLift', 'tc7_houseType', 'tc7_isActivity',
                   'tc7_isAiLock', 'tc7_isBalcony', 'tc7_isCz', 'tc7_isDw', 'tc7_isHighRate', 'tc7_isNew',
                   'tc7_isSubsidy', 'tc7_priceRange', 'tc7_styleCode', 'tc7_supplyHeat', 'tc7_cityCode', 'tc7_isTop',
                   'tc7_isButtom', 'tc7_canPet', 'tc7_isDeepBreath', 'tc7_isDiscount',
                   'tc14_airQuality', 'tc14_area', 'tc14_bedroomType', 'tc14_buildYear', 'tc14_compartmentFace',
                   'tc14_disposeBedroomAmount', 'tc14_floor', 'tc14_haveLift', 'tc14_houseType', 'tc14_isActivity',
                   'tc14_isAiLock', 'tc14_isBalcony', 'tc14_isCz', 'tc14_isDw', 'tc14_isHighRate', 'tc14_isNew',
                   'tc14_isSubsidy', 'tc14_priceRange', 'tc14_styleCode', 'tc14_supplyHeat', 'tc14_cityCode',
                   'tc14_isTop', 'tc14_isButtom', 'tc14_canPet', 'tc14_isDeepBreath', 'tc14_isDiscount',

                   'mulSubwayStation', 'mulSubwayLine', 'mulBizcirclecode',
                   'ta3_mulResblockName', 'ta3_mulDistrictCode', 'ta3_mulBizcirclecode', 'ta3_mulSubwayStation',
                   'ta3_mulSubwayLine',
                   'ta7_mulResblockName', 'ta7_mulDistrictCode', 'ta7_mulBizcirclecode', 'ta7_mulSubwayStation',
                   'ta7_mulSubwayLine',
                   'ta14_mulResblockName', 'ta14_mulDistrictCode', 'ta14_mulBizcirclecode', 'ta14_mulSubwayStation',
                   'ta14_mulSubwayLine',

                   'tb3_mulResblockName', 'tb3_mulDistrictCode', 'tb3_mulBizcirclecode', 'tb3_mulSubwayStation',
                   'tb3_mulSubwayLine',
                   'tb7_mulResblockName', 'tb7_mulDistrictCode', 'tb7_mulBizcirclecode', 'tb7_mulSubwayStation',
                   'tb7_mulSubwayLine',
                   'tb14_mulResblockName', 'tb14_mulDistrictCode', 'tb14_mulBizcirclecode', 'tb14_mulSubwayStation',
                   'tb14_mulSubwayLine',

                   'tc3_mulResblockName', 'tc3_mulDistrictCode', 'tc3_mulBizcirclecode', 'tc3_mulSubwayStation',
                   'tc3_mulSubwayLine',
                   'tc7_mulResblockName', 'tc7_mulDistrictCode', 'tc7_mulBizcirclecode', 'tc7_mulSubwayStation',
                   'tc7_mulSubwayLine',
                   'tc14_mulResblockName', 'tc14_mulDistrictCode', 'tc14_mulBizcirclecode', 'tc14_mulSubwayStation',
                   'tc14_mulSubwayLine', 'hid'

                   ]

dense_features = ['roomPrice', 'areaNumber', 'house_ctr', 'user_ctr']

ctr = train_df['res'].values

cols = sparse_features + dense_features


def toint(x):
    return int(x)


def convert(x):
    return list(map(toint, x))


train_df['invnolist_arr'] = list(map(convert, list(x.split(',') for x in train_df['invnolist'].values.tolist())))
train_df['pricelist_arr'] = list(map(convert, list(x.split(',') for x in train_df['pricelist'].values.tolist())))
train_df['resblocklist_arr'] = list(map(convert, list(x.split(',') for x in train_df['resblocklist'].values.tolist())))
train_df['stationlist_arr'] = list(map(convert, list(x.split(',') for x in train_df['stationlist'].values.tolist())))
train_df['linelist_arr'] = list(map(convert, list(x.split(',') for x in train_df['linelist'].values.tolist())))

mul_field_map = {}
mul_field_map["invnolist_arr"] = np.array(train_df["invnolist_arr"].values.tolist())
mul_field_map["pricelist_arr"] = np.array(train_df["pricelist_arr"].values.tolist())
mul_field_map["resblocklist_arr"] = np.array(train_df["resblocklist_arr"].values.tolist())
mul_field_map["stationlist_arr"] = np.array(train_df["stationlist_arr"].values.tolist())
mul_field_map["linelist_arr"] = np.array(train_df["linelist_arr"].values.tolist())

train_model_input = {name: train_df[name].values for name in cols}
train_model_input.update(mul_field_map)

field_input = []


def get_behavior_embedding():
    invnolist_arr = tf.keras.layers.Input(shape=[20], name="invnolist_arr")
    field_input.append(invnolist_arr)
    invno_his_emb = tf.keras.layers.Embedding(train_df["hid"].max() + 30000, 8, mask_zero=True,
                                              name="invnolist_arr_emb")(invnolist_arr)

    price_arr = tf.keras.layers.Input(shape=[20], name="pricelist_arr")
    field_input.append(price_arr)
    price_his_emb = tf.keras.layers.Embedding(train_df["priceRange"].max() + 30000, 8, mask_zero=True,
                                              name="pricelist_arr_emb")(price_arr)  # None * 3 * K

    resblock_arr = tf.keras.layers.Input(shape=[20], name="resblocklist_arr")
    field_input.append(resblock_arr)
    resblock_his_emb = tf.keras.layers.Embedding(train_df["resblockName"].max() + 30000, 8, mask_zero=True,
                                                 name="resblocklist_arr_emb")(resblock_arr)  # None * 3 * K

    station_arr = tf.keras.layers.Input(shape=[20], name="stationlist_arr")
    field_input.append(station_arr)
    station_his_emb = tf.keras.layers.Embedding(train_df["mulSubwayStation"].max() + 30000, 8, mask_zero=True,
                                                name="stationlist_arr_emb")(station_arr)  # None * 3 * K

    line_arr = tf.keras.layers.Input(shape=[20], name="linelist_arr")
    field_input.append(line_arr)
    line_his_emb = tf.keras.layers.Embedding(train_df["mulSubwayLine"].max() + 30000, 8, mask_zero=True,
                                             name="linelist_arr_emb")(line_arr)  # None * 3 * K
    behvr_emb = tf.concat([invno_his_emb, price_his_emb, resblock_his_emb, station_his_emb, line_his_emb],
                          -1)  # shape(batch_size, max_seq_len, embedding_size)
    return invnolist_arr, behvr_emb


def attention(inputs, context, masks):
    with tf.variable_scope("attention_layer"):
        shape = inputs.shape.as_list()
        max_seq_len = shape[1]  # padded_dim
        embedding_size = shape[2]
        seq_emb = tf.reshape(inputs, shape=[-1, embedding_size])  # shape(batch_size * max_seq_len, embedding_size)
        ctx_len = context.shape.as_list()[1]
        ctx_emb = tf.reshape(tf.tile(context, [1, max_seq_len]), shape=[-1, ctx_len])
        print("seq_emb shape:", seq_emb.shape)
        print("ctx_emb shape:", ctx_emb.shape)
        net = tf.concat([seq_emb, ctx_emb], axis=1)
        print("attention input shape:", net.shape)
        for units in [256,128, 64]:
            net = tf.keras.layers.Dense(int(units))(net)
            net = tf.keras.layers.BatchNormalization()(net)
            net = tf.keras.layers.Activation('relu')(net)

        net = tf.keras.layers.Dense(1)(net)
        att_wgt = tf.keras.layers.Activation('sigmoid')(net)
        att_wgt = tf.reshape(att_wgt, shape=[-1, max_seq_len, 1], name="weight")  # shape(batch_size, max_seq_len, 1)
        wgt_emb = tf.multiply(inputs, att_wgt)  # shape(batch_size, max_seq_len, embedding_size)
        masks = tf.expand_dims(masks, axis=-1)
        att_emb = tf.reduce_sum(tf.multiply(wgt_emb, masks), 1, name="weighted_embedding")
        return att_emb


def base_model():
    field_embedding = []
    dense_input = []
    for cat in cols:
        if cat in sparse_features:
            input = tf.keras.layers.Input(shape=(1,), name=cat, dtype="int32")
            field_input.append(input)
            nums = train_df[cat].max() + 3
            # ffm embeddings
            x_embed = tf.keras.layers.Embedding(nums, 8, input_length=1, trainable=True,
                                     embeddings_regularizer=regularizers.l2(1e-5), name=cat + "_emb")(input)
            x_reshape = tf.keras.layers.Reshape((8,))(x_embed)
            field_embedding.append(x_reshape)
        else:
            dense = tf.keras.layers.Input(shape=[1], name=cat)  # None*1
            field_input.append(dense)
            dense_input.append(dense)

    mask, behvr_emb = get_behavior_embedding()

    # lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=8)
    # outputs, state = tf.nn.dynamic_rnn(lstm_cell, behvr_emb, dtype=tf.float32)
    # print("lstm output shape:", outputs.shape)

    masks = tf.cast(mask > 0, tf.float32)
    context = tf.keras.layers.concatenate(field_embedding + dense_input, axis=-1)
    print("attention context shape:", context.shape)
    sequence = attention(behvr_emb, context, masks)
    print("sequence embedding shape:", sequence.shape)

    net = tf.concat([sequence, context], -1)

    embed_layer = tf.keras.layers.Dense(256)(net)
    embed_layer = tf.keras.layers.BatchNormalization()(embed_layer)
    embed_layer = tf.keras.layers.Activation('relu')(embed_layer)

    embed_layer = tf.keras.layers.Dense(128)(embed_layer)
    embed_layer = tf.keras.layers.BatchNormalization()(embed_layer)
    embed_layer = tf.keras.layers.Activation('relu')(embed_layer)

    embed_layer = tf.keras.layers.Dense(64)(embed_layer)
    embed_layer = tf.keras.layers.BatchNormalization()(embed_layer)
    embed_layer = tf.keras.layers.Activation('relu')(embed_layer)

    embed_layer = tf.keras.layers.Dense(1)(embed_layer)

    ctr_predictions = tf.keras.layers.Activation('sigmoid', name="ctr")(embed_layer)

    ctcvr_model = tf.keras.Model(
        inputs=field_input,
        outputs=ctr_predictions)

    # plot_model(ctcvr_model, 'model.png', show_shapes=True)

    opt = tf.keras.optimizers.Adam(0.001)
    ctcvr_model.compile(optimizer=opt, loss="binary_crossentropy",
                        metrics=[auc])
    return ctcvr_model


# training########################################

# es = EarlyStopping(monitor='val_auc',mode='max',patience=3)
# checkpoint = ModelCheckpoint( file_path, save_weights_only=True, verbose=1, save_best_only=True)
model = base_model()
model.fit(train_model_input, ctr,
          batch_size=10000,
          epochs=20,
          validation_split=0.2, verbose=2, shuffle=True, callbacks=[EarlyStoppingAtMinLoss(3)])

#model.save('dupn_model.h5')
#test-------------------------------
test_df = pd.read_csv("/data/home/wanghk/datadin/din.csv", sep=';', dtype=date_type)
test_df['invnolist_arr'] = list(map(convert, list(x.split(',') for x in test_df['invnolist'].values.tolist())))
test_df['pricelist_arr'] = list(map(convert, list(x.split(',') for x in test_df['pricelist'].values.tolist())))
test_df['resblocklist_arr'] = list(map(convert, list(x.split(',') for x in test_df['resblocklist'].values.tolist())))
test_df['stationlist_arr'] = list(map(convert, list(x.split(',') for x in test_df['stationlist'].values.tolist())))
test_df['linelist_arr'] = list(map(convert, list(x.split(',') for x in test_df['linelist'].values.tolist())))

mul_field_map_test = {}
mul_field_map_test["invnolist_arr"] = np.array(test_df["invnolist_arr"].values.tolist())
mul_field_map_test["pricelist_arr"] = np.array(test_df["pricelist_arr"].values.tolist())
mul_field_map_test["resblocklist_arr"] = np.array(test_df["resblocklist_arr"].values.tolist())
mul_field_map_test["stationlist_arr"] = np.array(test_df["stationlist_arr"].values.tolist())
mul_field_map_test["linelist_arr"] = np.array(test_df["linelist_arr"].values.tolist())

test_model_input = {name: test_df[name].values for name in cols}
test_model_input.update(mul_field_map_test)

pred_ans = model.predict(test_model_input, batch_size=1000)
print(pred_ans)
print("test ctr-AUC", round(roc_auc_score(test_df["res"].values, pred_ans), 4))
print("end:", datetime.datetime.now())
