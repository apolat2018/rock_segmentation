# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 14:05:52 2021

@author: ap
"""
import os,sys
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import segmentation_models as sm
import cv2
import tensorflow.keras as keras
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import albumentations as A
import pickle
from statistics import mean
import datetime
import random
sm.set_framework('tf.keras')
sm.framework()
print("sm: ",sm.__version__,"keras:",keras.__version__,"tf: ",tf.__version__)

DATA_DIR = 'D:\\rocksegmentation\\random_sampling_dataset_256'
height=256
width=256

x_train_dir = os.path.join(DATA_DIR, 'train')
y_train_dir = os.path.join(DATA_DIR, 'train_mask')

x_valid_dir = os.path.join(DATA_DIR, 'val')
y_valid_dir = os.path.join(DATA_DIR, 'val_mask')

#x_test_dir = os.path.join(DATA_DIR, 'test')
#y_test_dir = os.path.join(DATA_DIR, 'test_mask')

# helper function for data visualization
def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image,cmap="gray")
    #plt.savefig("deneme.png"),
    plt.show()
    
# helper function for data visualization    
def denormalize(x):
    """Scale image to range 0..1 for correct plot"""
    x_max = np.percentile(x, 98)
    x_min = np.percentile(x, 2)    
    x = (x - x_min) / (x_max - x_min)
    x = x.clip(0, 1)
    return x
    

# classes for data loading and preprocessing
class Dataset:
    
    CLASSES = ["background","rock"]
    
    def __init__(
            self, 
            images_dir, 
            masks_dir, 
            classes=None, 
            augmentation=None, 
            preprocessing=None,
    ):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
        
        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        print(type(preprocessing))
    
    def __getitem__(self, i):
        
        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
       
        mask = cv2.imread(self.masks_fps[i], 0)
        mask = np.where(mask>0,1,0)#bu ra benim 
        
        # extract certain classes from mask (e.g. cars)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')
        
        
        # add background if mask is not binary
        if mask.shape[-1] != 1:
            background = 1 - mask.sum(axis=-1, keepdims=True)
            mask = np.concatenate((mask, background), axis=-1)
            
        
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            
        return image, mask
        
    def __len__(self):
        return len(self.ids)
    
    
class Dataloder(keras.utils.Sequence):
    
    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(dataset))

        self.on_epoch_end()

    def __getitem__(self, i):
        
        # collect batch data
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        data = []
        for j in range(start, stop):
            data.append(self.dataset[j])
        
        # transpose list of lists
        batch = [np.stack(samples, axis=0) for samples in zip(*data)]
        
        return tuple (batch)#tuple sonradan eklendi   
    
    def __len__(self):
        """Denotes the number of batches per epoch"""
        return len(self.indexes) // self.batch_size
    
    def on_epoch_end(self):
        """Callback function to shuffle indexes each epoch"""
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)

# Lets look at data we have
dataset = Dataset(x_train_dir, y_train_dir, classes=["background",'rock'])
r=random.randint(0, len(dataset))
print(len(dataset))
image, mask = dataset[r] # get some sample
visualize(
    image=image, 
    back_ground=mask[...,0].squeeze(),
   
    rock=mask[...,1].squeeze(),
    
)

def round_clip_0_1(x, **kwargs):
    return x.round().clip(0, 1)

# define heavy augmentations
def get_training_augmentation():
    train_transform = [

        A.HorizontalFlip(p=0.5),

        A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),

        A.PadIfNeeded(min_height=height, min_width=width, always_apply=True, border_mode=0),
        A.RandomCrop(height=height, width=width, always_apply=True),

        #A.IAAAdditiveGaussianNoise(p=0.2),
        #A.IAAPerspective(p=0.5),
        A.GaussNoise(p=0.5),
        A.Perspective(p=0.5),

        A.OneOf(
            [
                A.CLAHE(p=1),
                A.RandomBrightness(p=1),
                A.RandomGamma(p=1),
            ],
            p=0.9,
        ),

        A.OneOf(
            [
                #A.IAASharpen(p=1),
                A.Sharpen(p=1),
                A.Blur(blur_limit=3, p=1),
                A.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),

        A.OneOf(
            [
                A.RandomContrast(p=1),
                A.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
        A.Lambda(mask=round_clip_0_1)
    ]
    return A.Compose(train_transform)


def get_validation_augmentation():
 
    test_transform = [
        A.PadIfNeeded(height, width)
    ]
    return A.Compose(test_transform)

def get_preprocessing(preprocessing_fn):

    
    _transform = [
        A.Lambda(image=preprocessing_fn),
    ]
    return A.Compose(_transform)

# Lets look at augmented data we have
dataset = Dataset(x_train_dir, y_train_dir, classes=["background",'rock'], augmentation=get_training_augmentation())
print(len(dataset))
image, mask = dataset[11] # get some sample
visualize(
    image=image, 
    dolins_mask=mask.squeeze()
   
)

BACKBONE = 'resnext101'
#'densenet121'fenadeğil
#'resnet101'
#'resnet34'
#'efficientnetb6'
#'inceptionresnetv2'
#"resnext101"

BATCH_SIZE = 8
CLASSES = ['rock']
LR = 0.0001 #0.0001 best for adam
EPOCHS = 100
#Training with n fold cross correletaions
cc=1
print("/".join(DATA_DIR.split("/")[:-1]))

check=0
save_path=os.path.join("/".join(DATA_DIR.split("/")[:-1]),BACKBONE+"result")
if not os.path.exists(save_path):
  os.mkdir(save_path)
for i in range(cc):
  preprocess_input = sm.get_preprocessing(BACKBONE)
  # define network parameters
  n_classes = 1 if len(CLASSES) == 1 else (len(CLASSES) + 1)  # case for binary and multiclass segmentation
  activation = 'sigmoid' if n_classes == 1 else 'softmax'
  #create model
  model = sm.Unet(BACKBONE, classes=n_classes, activation=activation)
  # define optomizer

  optim = keras.optimizers.Adam(LR)
  #optim = keras.optimizers.SGD(LR)
  #optim = keras.optimizers.RMSprop(LR)

  # Segmentation models losses can be combined together by '+' and scaled by integer or float factor
  loss_deneme=sm.losses.binary_focal_dice_loss
  dice_loss = sm.losses.DiceLoss() 
  focal_loss = sm.losses.BinaryFocalLoss() if n_classes == 1 else sm.losses.CategoricalFocalLoss()
  total_loss = dice_loss + (1 * focal_loss)

  # actulally total_loss can be imported directly from library, above example just show you how to manipulate with losses
  #total_loss = sm.losses.binary_focal_dice_loss # or sm.losses.categorical_focal_dice_loss 

  metrics = [sm.metrics.IOUScore(threshold=0.5,class_indexes=0), sm.metrics.FScore(threshold=0.5)] #class_index=0 yapıldı
  metrics_deneme=[sm.metrics.IOUScore(threshold=0.5,class_indexes=0)]

# compile keras model with defined optimozer, loss and metrics
  model.compile(optim, loss_deneme, metrics)
  # Dataset for train images
  train_dataset = Dataset(
      x_train_dir, 
      y_train_dir, 
      classes=CLASSES, 
      augmentation=get_training_augmentation(),
      preprocessing=get_preprocessing(preprocess_input),
  )

  # Dataset for validation images
  valid_dataset = Dataset(
      x_valid_dir, 
      y_valid_dir, 
      classes=CLASSES, 
      augmentation=get_validation_augmentation(),
      preprocessing=get_preprocessing(preprocess_input),
  )

  train_dataloader = Dataloder(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
  valid_dataloader = Dataloder(valid_dataset, batch_size=1, shuffle=False)

  # check shapes for errors
  assert train_dataloader[0][0].shape == (BATCH_SIZE, height, width, 3)
  assert train_dataloader[0][1].shape == (BATCH_SIZE, height, width, n_classes)
  log_dir="/content/drive/MyDrive/Dolin_Segmentation/logs"
  # define callbacks for learning rate scheduling and best checkpoints saving
  #keras.callbacks.ModelCheckpoint('/content/drive/MyDrive/Dolin_Segmentation/best_model.h5', save_weights_only=True, save_best_only=True, mode='min'),
  callbacks = [keras.callbacks.ReduceLROnPlateau(monitor="val_loss",factor=0.1,patience=10,verbose=1),
  ]
 
  history = model.fit_generator(
    train_dataloader, 
    steps_per_epoch=len(train_dataloader), 
    epochs=EPOCHS, 
    callbacks=callbacks, 
    validation_data=valid_dataloader, 
    validation_steps=len(valid_dataloader),
  )
  
  scores = model.evaluate_generator(valid_dataloader)
  val_iou=scores[1]
#--------------------save as txt------------
  f=open(os.path.join(save_path,"rslts.txt"),"a+")
  f.write("Tarih zaman= "+str(datetime.datetime.now())+"\n"
          "Data directory= "+DATA_DIR+"\n"
          "Model= "+BACKBONE+"\n"
          "epoch= "+str(EPOCHS)+"\n"
          "Batch size= "+str(BATCH_SIZE)+"\n"
          "image size= "+str(height)+"x"+str(width)+"\n"
          "Optimizer= "+str(optim)+"\n"
          "Learning Rate= "+str(LR)+"\n"
          "Validation iou = "+str(scores[1])+"\n"
          "Validation F1 = "+str(scores[2])+"\n"
          "Train iou = "+str(history.history['iou_score'][-1])+"\n"
          "Train F1 = "+str(history.history['f1-score'][-1])+"\n"
        #"LR= "+str(history.history["lr"])+"\n"
        "----------------------------------------------------------------------\n")
  f.close()
#-----------------------save best model and history--------------------------
  if val_iou>check:

    # Plot training & validation iou_score values
    plt.figure(figsize=(10, 10))
    plt.subplot(211)
    plt.plot(history.history['iou_score'])
    plt.plot(history.history['val_iou_score'])
    plt.title('Model iou_score')
    plt.ylabel('iou_score')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    # Plot training & validation loss values
    plt.subplot(212)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper right')
    plt.savefig(os.path.join(save_path,"rslts.png"))
    plt.show()
    check=val_iou
    
    #model.save_weights(os.path.join(save_path,"best_model.h5"))
    model.save(os.path.join(save_path,"best_model_tflite.h5"))
    #model.save(os.path.join(save_path,"best_model.h5"))
    with open(os.path.join(save_path,"history"), 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
  
    #del model
    print("---------------------------------------------------------CC Number = " +str(i+1)+ "  finished-----------------------------------------")   
  else:
    pass
    print("---------------------------------------------------------CC Number = " +str(i+1)+ "  finished-----------------------------------------")
    
    #del model
   

import random

def random_predict(valid_dataset):
    n = 1
    ids = np.random.choice(np.arange(len(valid_dataset)), size=n)
    
    
    
    for i in ids:
        
        image, gt_mask = valid_dataset[i]
        image = np.expand_dims(image, axis=0)
        pr_mask = model.predict(image).round()#segmentation için >0.5
        
        print("maximum value:",np.max(pr_mask))
        print(np.unique(pr_mask))
        
    
    
        img=pr_mask
        image=denormalize(image.squeeze())
        
    
        img = np.reshape(pr_mask,(height,width)).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)#gürültüleri temizlemek için
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)#gürültüleri temizlemek için iç gürültü
        
       
        print(np.unique(img))
        #gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        edge = cv2.Canny(img, 0, 1)
        print("edge",np.unique(edge))
    
        contours = cv2.findContours(edge.copy(), 
                                cv2.RETR_LIST,
                                cv2.CHAIN_APPROX_SIMPLE)
      
        
        """for cnt in contours[0]:
          (x,y),radius = cv2.minEnclosingCircle(cnt)
          center = (int(x),int(y))
          radius = int(radius)
          cv2.circle(image,center,radius,(0,255,0),2)"""
        #cv2.drawContours(image, contours[0], -1, (0,255,0), thickness = 2)
        rslts=image.copy()
        #rslts=np.zeros((512,512,3),dtype=np.float32)
        cv2.polylines(rslts,contours[0],True,(0,255,0),2)
    
    
        visualize(
            image=denormalize(image.squeeze()),
            gt_mask=gt_mask[..., 0].squeeze(),
            pr_mask=pr_mask[..., 0].squeeze(),
            Result=rslts)
       
        """# IoU calculation
        result1=gt_mask=gt_mask[..., 0].squeeze()
        result2=pr_mask=pr_mask[..., 0].squeeze()
        intersection = np.logical_and(result1, result2)
        union = np.logical_or(result1, result2)
        iou_score = np.sum(intersection) / np.sum(union)
        print("IoU is %s" % iou_score)
        visualize(kesisism=intersection,
                  birlesim=union)
        #plt.imshow(image)
        #plt.show()"""
for i in range(3):
    random_predict(valid_dataset)
