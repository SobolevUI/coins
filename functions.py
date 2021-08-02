import numpy as np
import os
import cv2
from PIL import Image
import io
# import shutil
import pandas as pd
from tensorflow.keras.layers import Dense, Flatten, Reshape, Input, Conv2DTranspose, concatenate, Activation, \
    MaxPooling2D, Conv2D, BatchNormalization, UpSampling2D, Dropout, GlobalAveragePooling2D, UpSampling2D
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import decode_predictions, preprocess_input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Flatten, Input, GlobalAveragePooling2D, GlobalMaxPooling2D
from tensorflow.keras.layers import Dense, Flatten, Reshape, Input, Conv2DTranspose, concatenate, Activation, \
    MaxPooling2D, Conv2D, BatchNormalization
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from glob import glob
import pandas as pd
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from scipy.spatial import distance
import pickle
from  threading import Thread
import queue
folder = 'C:/Users/Юрий/PycharmProjects/pythonProject1/src/images'
def load_image(path):
    # img = Image.open(io.BytesIO(path))
    # img = Image.open(path)
    # if img.mode != 'RGB': img = img.convert('RGB')
    # img = img.resize((224, 224))
    # x = image.img_to_array(img)
    # x = np.expand_dims(x, axis=0)
    # x = preprocess_input(x)
    x=cv2.resize(cv2.imread(path, cv2.IMREAD_COLOR), (224, 224))
    # x=preprocess_input(x).reshape(1,224,224,3)/255
    return x
def contrast(img, contrast):
    clahe = cv2.createCLAHE(clipLimit=contrast, tileGridSize=(5,5))
    return clahe.apply(img)
def get_edges(img):
    img = np.uint8(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    img2 = contrast(img, 0.5)
    img3 = contrast(img, 1)
    return np.stack([img, img2, img3], axis=2)

def create_emb(folder,model,coin=0):
  test_features = np.zeros((2, 25088))
  images=[]
  for i,filename in enumerate(os.listdir(folder)):
    file_path = os.path.join(folder, filename)
    test_features=np.zeros((2,25088))
    print('file_path ',file_path)
    print('load_image(file_path))/255', load_image(file_path).shape)
    # print('model.predict(preprocess_input(load_image(file_path))/255)', (model.predict(preprocess_input(load_image(file_path))/255).shape))
    # test_features.append(model.predict(load_image(file_path)))
    img=load_image(file_path)
    if coin ==1:
        img=get_edges(img)
        img = preprocess_input(img).reshape(1, 224, 224, 3)
    else:
        img=preprocess_input(img).reshape(1,224,224,3)/255
    test_features[i]=model.predict(img)
    images.append(os.path.join(folder, filename))
    print('images from create_emb ',images)

  # test_features=np.array(test_features)
  print('test_features.shape ', test_features.shape)
  return test_features,images

def combine_embs(model,features,coin=0,path=None,ind=None,train_path=None):
  if path:
    a=create_emb(path,model,coin)[0]
    # a=np.sum([a[0],a[1]],axis=0)
    a=a[0]+a[1]
    print('np.sum([a[0],a[1]]', a.shape)
    a=np.expand_dims(a,axis=0)
    features = np.concatenate([features, a])
    return features
  else:
      a = []
      img_a_path = train_path + str(ind) + 'a.jpg'
      img_r_path = train_path + str(ind) + 'r.jpg'
      img_a = preprocess_input(load_image(img_a_path)).reshape(1, 224, 224, 3) / 255
      img_r = preprocess_input(load_image(img_r_path)).reshape(1, 224, 224, 3) / 255
      a.append(model.predict(img_a))
      a.append(model.predict(img_r))
      a = np.sum([a[0], a[1]], axis=0)
      # a = np.expand_dims(a, axis=0)
      return a
      # features = np.concatenate([features, a])
  # return features

def project_pca(path,model,pca,features):
  a=combine_embs(path,model)
  pca_test = pca.transform(a)
  pca_features=np.concatenate([features,pca_test])
  return pca_features

def srav(dd1, d, out):
  distances = [distance.cosine(d, feat) for feat in dd1]
  out.put(distances)

# def get_closest_images(pca_features, num_results=10):
def get_closest_images(features, num_results=10):
# def get_closest_images(dd, d_in, delta = 10000, num_results=10):
    # pca_features = df.iloc[:,:-1].values
    # result = []
    # out_all = queue.Queue()
    # ths = [Thread(target=srav, args=(dd[i - delta: i], d_in, out_all)) for i in range(delta, dd.shape[0] + 1, delta)]
    # for x in ths:
    #     x.start()
    #     result.extend(out_all.get())
    # for tr in ths:
    #     tr.join()
    # idx_closest = sorted(range(len(result)), key=lambda k: result[k])[1:num_results + 1]

   distances = [distance.cosine(features[-1], feat) for feat in features]
   idx_closest = sorted(range(len(distances)), key=lambda k: distances[k])[1:num_results+1]

   return idx_closest

def get_concatenated_images(indexes,test_images,train_path,df_total, thumb_height=200):
    thumbs = []
    for i in test_images:
      img=image.load_img(i)
      img=img.resize((int(img.width * thumb_height / img.height), thumb_height))
      thumbs.append(img)
    concat_image = np.concatenate([np.asarray(t) for t in thumbs], axis=1)
    concat_image = image.array_to_img(concat_image)
    concat_image.save('./images/000.jpg')
    print('test image')
    plt.figure(figsize = (10,15))
    plt.imshow(concat_image)
    plt.show()

    for idx in indexes:
        thumbs = []
        img_a_path=train_path+str(df_total.iloc[idx,0])+'a.jpg'
        img_r_path=train_path+str(df_total.iloc[idx,0])+'r.jpg'
        print('img_a_path ',img_a_path)
        print('img_r_path ', img_r_path)
        img_a = image.load_img(img_a_path)
        img_r = image.load_img(img_r_path)
        print('id = ',df_total.iloc[idx,0])
        img_a = img_a.resize((int(img_a.width * thumb_height / img_a.height), thumb_height))
        img_r = img_r.resize((int(img_r.width * thumb_height / img_r.height), thumb_height))
        thumbs.append(img_a)
        thumbs.append(img_r)
        concat_image = np.concatenate([np.asarray(t) for t in thumbs], axis=1)
        concat_image=image.array_to_img(concat_image)
        concat_image.save('./images/'+str(df_total.iloc[idx,0])+'.jpg')
        plt.figure(figsize = (10,15))
        plt.imshow(concat_image)
        plt.show()
    concat_image = np.concatenate([np.asarray(t) for t in thumbs], axis=1)
