
import efficientnet.tfkeras as efn
import tensorflow as tf

import os

import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut

from PIL import Image
import pandas as pd
from tqdm.auto import tqdm

df = pd.read_csv('../input/siim-covid19-detection/sample_submission.csv')
if df.shape[0] == 2477:
    fast_sub = True
    fast_df = pd.DataFrame(([['00086460a852_study', 'negative 1 0 0 1 1'], 
                         ['000c9c05fd14_study', 'negative 1 0 0 1 1'], 
                         ['65761e66de9f_image', 'none 1 0 0 1 1'], 
                         ['51759b5579bc_image', 'none 1 0 0 1 1']]), 
                       columns=['id', 'PredictionString'])
else:
    fast_sub = False





def read_xray(path, voi_lut = True, fix_monochrome = True):
    # Original from: https://www.kaggle.com/raddar/convert-dicom-to-np-array-the-correct-way
    dicom = pydicom.read_file(path)
    
    # VOI LUT (if available by DICOM device) is used to transform raw DICOM data to 
    # "human-friendly" view
    if voi_lut:
        data = apply_voi_lut(dicom.pixel_array, dicom)
    else:
        data = dicom.pixel_array
               
    # depending on this value, X-ray may look inverted - fix that:
    if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
        data = np.amax(data) - data
        
    data = data - np.min(data)
    data = data / np.max(data)
    data = (data * 255).astype(np.uint8)
        
    return data


def resize(array, size, keep_ratio=False, resample=Image.LANCZOS):
    # Original from: https://www.kaggle.com/xhlulu/vinbigdata-process-and-resize-to-image
    im = Image.fromarray(array)
    
    if keep_ratio:
        im.thumbnail((size, size), resample)
    else:
        im = im.resize((size, size), resample)
    
    return im


split = 'test'
save_dir = f'/kaggle/tmp/{split}/'

os.makedirs(save_dir, exist_ok=True)

save_dir = f'/kaggle/tmp/{split}/study/'
os.makedirs(save_dir, exist_ok=True)
if fast_sub:
    xray = read_xray('../input/siim-covid19-detection/train/00086460a852/9e8302230c91/65761e66de9f.dcm')
    im = resize(xray, size=600)  
    study = '00086460a852' + '_study.png'
    im.save(os.path.join(save_dir, study))
    xray = read_xray('../input/siim-covid19-detection/train/000c9c05fd14/e555410bd2cd/51759b5579bc.dcm')
    im = resize(xray, size=600)  
    study = '000c9c05fd14' + '_study.png'
    im.save(os.path.join(save_dir, study))
else:   
    for dirname, _, filenames in tqdm(os.walk(f'../input/siim-covid19-detection/{split}')):
        for file in filenames:
            # set keep_ratio=True to have original aspect ratio
            xray = read_xray(os.path.join(dirname, file))
            im = resize(xray, size=600)  
            study = dirname.split('/')[-2] + '_study.png'
            im.save(os.path.join(save_dir, study))

image_id = []
dim0 = []
dim1 = []
splits = []
save_dir = f'/kaggle/tmp/{split}/image/'
os.makedirs(save_dir, exist_ok=True)
if fast_sub:
    xray = read_xray('../input/siim-covid19-detection/train/00086460a852/9e8302230c91/65761e66de9f.dcm')
    im = resize(xray, size=512)  
    im.save(os.path.join(save_dir,'65761e66de9f_image.png'))
    image_id.append('65761e66de9f.dcm'.replace('.dcm', ''))
    dim0.append(xray.shape[0])
    dim1.append(xray.shape[1])
    splits.append(split)
    xray = read_xray('../input/siim-covid19-detection/train/000c9c05fd14/e555410bd2cd/51759b5579bc.dcm')
    im = resize(xray, size=512)  
    im.save(os.path.join(save_dir, '51759b5579bc_image.png'))
    image_id.append('51759b5579bc.dcm'.replace('.dcm', ''))
    dim0.append(xray.shape[0])
    dim1.append(xray.shape[1])
    splits.append(split)
else:
    for dirname, _, filenames in tqdm(os.walk(f'../input/siim-covid19-detection/{split}')):
        for file in filenames:
            # set keep_ratio=True to have original aspect ratio
            xray = read_xray(os.path.join(dirname, file))
            im = resize(xray, size=512)  
            im.save(os.path.join(save_dir, file.replace('.dcm', '_image.png')))
            image_id.append(file.replace('.dcm', ''))
            dim0.append(xray.shape[0])
            dim1.append(xray.shape[1])
            splits.append(split)
meta = pd.DataFrame.from_dict({'image_id': image_id, 'dim0': dim0, 'dim1': dim1, 'split': splits})

###STUDY PREDICT###

import numpy as np 
import pandas as pd
if fast_sub:
    df = fast_df.copy()
else:
    df = pd.read_csv('../input/siim-covid19-detection/sample_submission.csv')
id_laststr_list  = []
for i in range(df.shape[0]):
    id_laststr_list.append(df.loc[i,'id'][-1])
df['id_last_str'] = id_laststr_list

study_len = df[df['id_last_str'] == 'y'].shape[0]




######!pip install /kaggle/input/kerasapplications -q
######!pip install /kaggle/input/efficientnet-keras-source-code/ -q --no-deps



def auto_select_accelerator():
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.experimental.TPUStrategy(tpu)
        print("Running on TPU:", tpu.master())
    except ValueError:
        strategy = tf.distribute.get_strategy()
    print(f"Running on {strategy.num_replicas_in_sync} replicas")

    return strategy


def build_decoder(with_labels=True, target_size=(300, 300), ext='jpg'):
    def decode(path):
        file_bytes = tf.io.read_file(path)
        if ext == 'png':
            img = tf.image.decode_png(file_bytes, channels=3)
        elif ext in ['jpg', 'jpeg']:
            img = tf.image.decode_jpeg(file_bytes, channels=3)
        else:
            raise ValueError("Image extension not supported")

        img = tf.cast(img, tf.float32) / 255.0
        img = tf.image.resize(img, target_size)

        return img

    def decode_with_labels(path, label):
        return decode(path), label

    return decode_with_labels if with_labels else decode


def build_augmenter(with_labels=True):
    def augment(img):
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
        return img

    def augment_with_labels(img, label):
        return augment(img), label

    return augment_with_labels if with_labels else augment


def build_dataset(paths, labels=None, bsize=32, cache=True,
                  decode_fn=None, augment_fn=None,
                  augment=True, repeat=True, shuffle=1024, 
                  cache_dir=""):
    if cache_dir != "" and cache is True:
        os.makedirs(cache_dir, exist_ok=True)

    if decode_fn is None:
        decode_fn = build_decoder(labels is not None)

    if augment_fn is None:
        augment_fn = build_augmenter(labels is not None)

    AUTO = tf.data.experimental.AUTOTUNE
    slices = paths if labels is None else (paths, labels)

    dset = tf.data.Dataset.from_tensor_slices(slices)
    dset = dset.map(decode_fn, num_parallel_calls=AUTO)
    dset = dset.cache(cache_dir) if cache else dset
    dset = dset.map(augment_fn, num_parallel_calls=AUTO) if augment else dset
    dset = dset.repeat() if repeat else dset
    dset = dset.shuffle(shuffle) if shuffle else dset
    dset = dset.batch(bsize).prefetch(AUTO)

    return dset

#COMPETITION_NAME = "siim-cov19-test-img512-study-600"
strategy = auto_select_accelerator()
BATCH_SIZE = strategy.num_replicas_in_sync * 16

IMSIZE = (224, 240, 260, 300, 380, 456, 528, 600, 512)

#load_dir = f"/kaggle/input/{COMPETITION_NAME}/"
if fast_sub:
    sub_df = fast_df.copy()
else:
    sub_df = pd.read_csv('../input/siim-covid19-detection/sample_submission.csv')
sub_df = sub_df[:study_len]
test_paths = f'/kaggle/tmp/{split}/study/' + sub_df['id'] +'.png'

sub_df['negative'] = 0
sub_df['typical'] = 0
sub_df['indeterminate'] = 0
sub_df['atypical'] = 0


label_cols = sub_df.columns[2:]

test_decoder = build_decoder(with_labels=False, target_size=(IMSIZE[7], IMSIZE[7]), ext='png')
dtest = build_dataset(
    test_paths, bsize=BATCH_SIZE, repeat=False, 
    shuffle=False, augment=False, cache=False,
    decode_fn=test_decoder
)

with strategy.scope():
    
    models = []
    
    models0 = tf.keras.models.load_model(
        '../input/siim-covid19-efnb7-train-study/model0.h5'
    )
    models1 = tf.keras.models.load_model(
        '../input/siim-covid19-efnb7-train-study/model1.h5'
    )
    models2 = tf.keras.models.load_model(
        '../input/siim-covid19-efnb7-train-study/model2.h5'
    )
    models3 = tf.keras.models.load_model(
        '../input/siim-covid19-efnb7-train-study/model3.h5'
    )
    models4 = tf.keras.models.load_model(
        '../input/siim-covid19-efnb7-train-study/model4.h5'
    )
    
    models.append(models0)
    models.append(models1)
    models.append(models2)
    models.append(models3)
    models.append(models4)

    
    
    
sub_df[label_cols] = sum([model.predict(dtest, verbose=1) for model in models]) / len(models)



sub_df.columns = ['id', 'PredictionString1', 'negative', 'typical', 'indeterminate', 'atypical']
df = pd.merge(df, sub_df, on = 'id', how = 'left')


##STUDY STRING##


for i in range(study_len):
    negative = df.loc[i,'negative']
    typical = df.loc[i,'typical']
    indeterminate = df.loc[i,'indeterminate']
    atypical = df.loc[i,'atypical']
    df.loc[i, 'PredictionString'] = f'negative {negative} 0 0 1 1 typical {typical} 0 0 1 1 indeterminate {indeterminate} 0 0 1 1 atypical {atypical} 0 0 1 1'


df_study = df[['id', 'PredictionString']]

# 2 CLASS prediction

if fast_sub:
    sub_df = fast_df.copy()
else:
    sub_df = pd.read_csv('../input/siim-covid19-detection/sample_submission.csv')
sub_df = sub_df[study_len:]
test_paths = f'/kaggle/tmp/{split}/image/' + sub_df['id'] +'.png'
sub_df['none'] = 0

label_cols = sub_df.columns[2]

test_decoder = build_decoder(with_labels=False, target_size=(IMSIZE[8], IMSIZE[8]), ext='png')
dtest = build_dataset(
    test_paths, bsize=BATCH_SIZE, repeat=False, 
    shuffle=False, augment=False, cache=False,
    decode_fn=test_decoder
)

with strategy.scope():
    
    models = []
    
    models0 = tf.keras.models.load_model(
        '../input/siim-covid19-efnb7-train-fold0-5-2class/model0.h5'
    )
    models1 = tf.keras.models.load_model(
        '../input/siim-covid19-efnb7-train-fold0-5-2class/model1.h5'
    )
    models2 = tf.keras.models.load_model(
        '../input/siim-covid19-efnb7-train-fold0-5-2class/model2.h5'
    )
    models3 = tf.keras.models.load_model(
        '../input/siim-covid19-efnb7-train-fold0-5-2class/model3.h5'
    )
    models4 = tf.keras.models.load_model(
        '../input/siim-covid19-efnb7-train-fold0-5-2class/model4.h5'
    )
    
    models.append(models0)
    models.append(models1)
    models.append(models2)
    models.append(models3)
    models.append(models4)

    
    
    
sub_df[label_cols] = sum([model.predict(dtest, verbose=1) for model in models]) / len(models)
df_2class = sub_df.reset_index(drop=True)

del models
del models0, models1, models2, models3, models4


#YOLOV5 PREDICTION#

import numpy as np, pandas as pd
from glob import glob
import shutil, os
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from tqdm.notebook import tqdm
import seaborn as sns
import torch


meta = meta[meta['split'] == 'test']
if fast_sub:
    test_df = fast_df.copy()
else:
    test_df = pd.read_csv('../input/siim-covid19-detection/sample_submission.csv')
test_df = df[study_len:].reset_index(drop=True) 
meta['image_id'] = meta['image_id'] + '_image'
meta.columns = ['id', 'dim0', 'dim1', 'split']
test_df = pd.merge(test_df, meta, on = 'id', how = 'left')




dim = 512 #1024, 256, 'original'
test_dir = f'/kaggle/tmp/{split}/image'
weights_dir = '/kaggle/input/siim-cov19-yolov5-train/yolov5/runs/train/exp/weights/best.pt'

shutil.copytree('/kaggle/input/yolov5-official-v31-dataset/yolov5', '/kaggle/working/yolov5')
os.chdir('/kaggle/working/yolov5') # install dependencies

import torch
#from IPython.display import Image, clear_output  # to display images

#clear_output()
#print('Setup complete. Using torch %s %s' % (torch.__version__, torch.cuda.get_device_properties(0) if torch.cuda.is_available() else 'CPU'))


!python detect.py --weights $weights_dir\
--img 512\
--conf 0.001\
--iou 0.5\
--source $test_dir\
--save-txt --save-conf --exist-ok
def yolo2voc(image_height, image_width, bboxes):
    """
    yolo => [xmid, ymid, w, h] (normalized)
    voc  => [x1, y1, x2, y1]

    """ 
    bboxes = bboxes.copy().astype(float) # otherwise all value will be 0 as voc_pascal dtype is np.int

    bboxes[..., [0, 2]] = bboxes[..., [0, 2]]* image_width
    bboxes[..., [1, 3]] = bboxes[..., [1, 3]]* image_height

    bboxes[..., [0, 1]] = bboxes[..., [0, 1]] - bboxes[..., [2, 3]]/2
    bboxes[..., [2, 3]] = bboxes[..., [0, 1]] + bboxes[..., [2, 3]]

    return bboxes
image_ids = []
PredictionStrings = []

for file_path in tqdm(glob('runs/detect/exp/labels/*.txt')):
    image_id = file_path.split('/')[-1].split('.')[0]
    w, h = test_df.loc[test_df.id==image_id,['dim1', 'dim0']].values[0]
    f = open(file_path, 'r')
    data = np.array(f.read().replace('\n', ' ').strip().split(' ')).astype(np.float32).reshape(-1, 6)
    data = data[:, [0, 5, 1, 2, 3, 4]]
    bboxes = list(np.round(np.concatenate((data[:, :2], np.round(yolo2voc(h, w, data[:, 2:]))), axis =1).reshape(-1), 12).astype(str))
    for idx in range(len(bboxes)):
        bboxes[idx] = str(int(float(bboxes[idx]))) if idx%6!=1 else bboxes[idx]
    image_ids.append(image_id)
    PredictionStrings.append(' '.join(bboxes))


pred_df = pd.DataFrame({'id':image_ids,
                        'PredictionString':PredictionStrings})






test_df = test_df.drop(['PredictionString'], axis=1)
sub_df = pd.merge(test_df, pred_df, on = 'id', how = 'left').fillna("none 1 0 0 1 1")
sub_df = sub_df[['id', 'PredictionString']]
for i in range(sub_df.shape[0]):
    if sub_df.loc[i,'PredictionString'] == "none 1 0 0 1 1":
        continue
    sub_df_split = sub_df.loc[i,'PredictionString'].split()
    sub_df_list = []
    for j in range(int(len(sub_df_split) / 6)):
        sub_df_list.append('opacity')
        sub_df_list.append(sub_df_split[6 * j + 1])
        sub_df_list.append(sub_df_split[6 * j + 2])
        sub_df_list.append(sub_df_split[6 * j + 3])
        sub_df_list.append(sub_df_split[6 * j + 4])
        sub_df_list.append(sub_df_split[6 * j + 5])
    sub_df.loc[i,'PredictionString'] = ' '.join(sub_df_list)
sub_df['none'] = df_2class['none'] 
for i in range(sub_df.shape[0]):
    if sub_df.loc[i,'PredictionString'] != 'none 1 0 0 1 1':
        sub_df.loc[i,'PredictionString'] = sub_df.loc[i,'PredictionString'] + ' none ' + str(sub_df.loc[i,'none']) + ' 0 0 1 1'
sub_df = sub_df[['id', 'PredictionString']]   
df_study = df_study[:study_len]
df_study = df_study.append(sub_df).reset_index(drop=True)
df_study.to_csv('/kaggle/working/submission.csv',index = False)  
shutil.rmtree('/kaggle/working/yolov5')



                        