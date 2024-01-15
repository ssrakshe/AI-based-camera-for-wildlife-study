fromtensorflow.keras.preprocessing.imageimportImageDataGenerator
from tensorflow.keras.preprocessing import image
fromtensorflow.keras.optimizersimport RMSprop
importtensorflowastf
importnumpyasnp
importmatplotlib.pyplotasplt
importcv2
importos
In [2]:
train =ImageDataGenerator(rescale=1/255)
test =ImageDataGenerator(rescale=1/255)
In [3]:
train_dataset=train.flow_from_directory(r'C:/Users/SHUBHAM/Documents/project2021/Wildlife4/New_Data_set/train/',target_size=(200,200),batch_size=5,class_mode='categorical')
test_dataset=train.flow_from_directory(r'C:/Users/SHUBHAM/Documents/project2021/Wildlife4/New_Data_set/test/',target_size=(200,200),batch_size=5,class_mode='categorical')
Found 156 images belonging to 6 classes.
Found 104 images belonging to 6 classes.
In [4]:
train_dataset.class_indices
Out[4]:
{'L24': 0, 'L25': 1, 'L54': 2, 'L61': 3, 'L66': 4, 'L82': 5}
In [5]:
test_dataset.classes
Out[5]:
array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
       2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
       3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5])
In [6]:
model=tf.keras.models.Sequential([tf.keras.layers.Conv2D(16,(3,3),activation='relu',input_shape=(200,200,3)),
#
tf.keras.layers.Conv2D(32,(3,3),activation='relu'),
tf.keras.layers.MaxPool2D(2,2),
tf.keras.layers.Conv2D(32,(3,3),activation='relu'),
tf.keras.layers.MaxPool2D(2,2),
tf.keras.layers.Conv2D(32,(3,3),activation='relu'),
tf.keras.layers.MaxPool2D(2,2),


##
tf.keras.layers.Flatten(),
##
tf.keras.layers.Dense(128,activation='relu'),
##
tf.keras.layers.Dense(6,activation='softmax')
                                                         ])

In [7]:
model.compile(loss='categorical_crossentropy',optimizer=RMSprop(lr=0.001),metrics=['categorical_accuracy'])
In [8]:
model_fit=model.fit(train_dataset,validation_data=test_dataset,steps_per_epoch=10,epochs=50)
Train for 10 steps, validate for 21 steps
Epoch 1/50
10/10 [==============================] - 17s 2s/step - loss: 2.7262 - categorical_accuracy: 0.2400 - val_loss: 1.7288 - val_categorical_accuracy: 0.2885
Epoch 2/50
10/10 [==============================] - 20s 2s/step - loss: 1.8059 - categorical_accuracy: 0.1400 - val_loss: 1.7421 - val_categorical_accuracy: 0.1635
Epoch 3/50
10/10 [==============================] - 20s 2s/step - loss: 1.7602 - categorical_accuracy: 0.1522 - val_loss: 1.7058 - val_categorical_accuracy: 0.2404
Epoch 4/50
10/10 [==============================] - 21s 2s/step - loss: 1.7736 - categorical_accuracy: 0.2800 - val_loss: 1.6880 - val_categorical_accuracy: 0.3173
Epoch 5/50
10/10 [==============================] - 20s 2s/step - loss: 1.7119 - categorical_accuracy: 0.2800 - val_loss: 1.6936 - val_categorical_accuracy: 0.2788
Epoch 6/50
10/10 [==============================] - 21s 2s/step - loss: 1.6534 - categorical_accuracy: 0.2174 - val_loss: 1.6957 - val_categorical_accuracy: 0.3558
Epoch 7/50
10/10 [==============================] - 21s 2s/step - loss: 1.7418 - categorical_accuracy: 0.3000 - val_loss: 1.6458 - val_categorical_accuracy: 0.2596
Epoch 8/50
10/10 [==============================] - 21s 2s/step - loss: 1.5969 - categorical_accuracy: 0.2800 - val_loss: 1.5961 - val_categorical_accuracy: 0.3558
Epoch 9/50
10/10 [==============================] - 21s 2s/step - loss: 1.4932 - categorical_accuracy: 0.3000 - val_loss: 1.7378 - val_categorical_accuracy: 0.2212
Epoch 10/50
10/10 [==============================] - 21s 2s/step - loss: 1.5075 - categorical_accuracy: 0.4000 - val_loss: 1.5363 - val_categorical_accuracy: 0.3846
Epoch 11/50
10/10 [==============================] - 21s 2s/step - loss: 1.4015 - categorical_accuracy: 0.6600 - val_loss: 1.3630 - val_categorical_accuracy: 0.4712
Epoch 12/50
10/10 [==============================] - 21s 2s/step - loss: 1.3618 - categorical_accuracy: 0.5600 - val_loss: 1.3227 - val_categorical_accuracy: 0.5673
Epoch 13/50
10/10 [==============================] - 20s 2s/step - loss: 0.9590 - categorical_accuracy: 0.6600 - val_loss: 1.5259 - val_categorical_accuracy: 0.3654
Epoch 14/50
10/10 [==============================] - 20s 2s/step - loss: 0.8390 - categorical_accuracy: 0.6800 - val_loss: 1.2298 - val_categorical_accuracy: 0.5962
Epoch 15/50
10/10 [==============================] - 18s 2s/step - loss: 0.6404 - categorical_accuracy: 0.7600 - val_loss: 1.3089 - val_categorical_accuracy: 0.5096
Epoch 16/50
10/10 [==============================] - 18s 2s/step - loss: 0.6380 - categorical_accuracy: 0.8400 - val_loss: 1.2642 - val_categorical_accuracy: 0.5962
Epoch 17/50
10/10 [==============================] - 18s 2s/step - loss: 0.9807 - categorical_accuracy: 0.7174 - val_loss: 1.1544 - val_categorical_accuracy: 0.5577
Epoch 18/50
10/10 [==============================] - 19s 2s/step - loss: 0.5833 - categorical_accuracy: 0.8200 - val_loss: 1.1187 - val_categorical_accuracy: 0.7308
Epoch 19/50
10/10 [==============================] - 18s 2s/step - loss: 0.4716 - categorical_accuracy: 0.8400 - val_loss: 1.0847 - val_categorical_accuracy: 0.6731
Epoch 20/50
10/10 [==============================] - 19s 2s/step - loss: 0.5304 - categorical_accuracy: 0.8600 - val_loss: 1.1182 - val_categorical_accuracy: 0.6827
Epoch 21/50
10/10 [==============================] - 18s 2s/step - loss: 0.6223 - categorical_accuracy: 0.8696 - val_loss: 1.2599 - val_categorical_accuracy: 0.7212
Epoch 22/50
10/10 [==============================] - 19s 2s/step - loss: 0.4388 - categorical_accuracy: 0.9000 - val_loss: 1.1669 - val_categorical_accuracy: 0.7115
Epoch 23/50
10/10 [==============================] - 18s 2s/step - loss: 0.3216 - categorical_accuracy: 0.9348 - val_loss: 1.4807 - val_categorical_accuracy: 0.7115
Epoch 24/50
10/10 [==============================] - 19s 2s/step - loss: 0.4926 - categorical_accuracy: 0.8478 - val_loss: 1.4539 - val_categorical_accuracy: 0.6442
Epoch 25/50
10/10 [==============================] - 19s 2s/step - loss: 0.1939 - categorical_accuracy: 0.9800 - val_loss: 1.6685 - val_categorical_accuracy: 0.7404
Epoch 26/50
10/10 [==============================] - 20s 2s/step - loss: 0.1288 - categorical_accuracy: 0.9600 - val_loss: 1.4052 - val_categorical_accuracy: 0.7788
Epoch 27/50
10/10 [==============================] - 20s 2s/step - loss: 1.2993 - categorical_accuracy: 0.8478 - val_loss: 1.6457 - val_categorical_accuracy: 0.5769
Epoch 28/50
10/10 [==============================] - 20s 2s/step - loss: 0.4511 - categorical_accuracy: 0.9400 - val_loss: 1.0204 - val_categorical_accuracy: 0.7596
Epoch 29/50
10/10 [==============================] - 20s 2s/step - loss: 0.1239 - categorical_accuracy: 1.0000 - val_loss: 1.4089 - val_categorical_accuracy: 0.7596
Epoch 30/50
10/10 [==============================] - 19s 2s/step - loss: 0.0575 - categorical_accuracy: 0.9800 - val_loss: 1.5923 - val_categorical_accuracy: 0.7596
Epoch 31/50
10/10 [==============================] - 19s 2s/step - loss: 0.2096 - categorical_accuracy: 0.9130 - val_loss: 1.5567 - val_categorical_accuracy: 0.7885
Epoch 32/50
10/10 [==============================] - 19s 2s/step - loss: 0.3063 - categorical_accuracy: 0.9400 - val_loss: 1.5522 - val_categorical_accuracy: 0.7788
Epoch 33/50
10/10 [==============================] - 21s 2s/step - loss: 0.0896 - categorical_accuracy: 0.9800 - val_loss: 1.4809 - val_categorical_accuracy: 0.7500
Epoch 34/50
10/10 [==============================] - 19s 2s/step - loss: 0.1239 - categorical_accuracy: 0.9400 - val_loss: 1.8021 - val_categorical_accuracy: 0.7788
Epoch 35/50
10/10 [==============================] - 20s 2s/step - loss: 0.0164 - categorical_accuracy: 1.0000 - val_loss: 1.4837 - val_categorical_accuracy: 0.7596
Epoch 36/50
10/10 [==============================] - 20s 2s/step - loss: 0.0184 - categorical_accuracy: 1.0000 - val_loss: 2.5908 - val_categorical_accuracy: 0.7308
Epoch 37/50
10/10 [==============================] - 19s 2s/step - loss: 0.0139 - categorical_accuracy: 1.0000 - val_loss: 1.8985 - val_categorical_accuracy: 0.7981
Epoch 38/50
10/10 [==============================] - 18s 2s/step - loss: 0.6612 - categorical_accuracy: 0.9000 - val_loss: 1.5291 - val_categorical_accuracy: 0.8077
Epoch 39/50
10/10 [==============================] - 20s 2s/step - loss: 0.0398 - categorical_accuracy: 1.0000 - val_loss: 2.2433 - val_categorical_accuracy: 0.8596
Epoch 40/50
10/10 [==============================] - 20s 2s/step - loss: 0.0114 - categorical_accuracy: 1.0000 - val_loss: 2.4632 - val_categorical_accuracy: 0.8692
Epoch 41/50
10/10 [==============================] - 20s 2s/step - loss: 0.0540 - categorical_accuracy: 0.9783 - val_loss: 2.1596 - val_categorical_accuracy: 0.8788
Epoch 42/50
10/10 [==============================] - 21s 2s/step - loss: 0.0589 - categorical_accuracy: 0.9600 - val_loss: 1.6624 - val_categorical_accuracy: 0.8596
Epoch 43/50
10/10 [==============================] - 19s 2s/step - loss: 0.0096 - categorical_accuracy: 1.0000 - val_loss: 1.8772 - val_categorical_accuracy: 0.8885
Epoch 44/50
10/10 [==============================] - 20s 2s/step - loss: 0.0036 - categorical_accuracy: 1.0000 - val_loss: 2.1454 - val_categorical_accuracy: 0.8981
Epoch 45/50
10/10 [==============================] - 20s 2s/step - loss: 1.8527 - categorical_accuracy: 0.8200 - val_loss: 1.3924 - val_categorical_accuracy: 0.8596
Epoch 46/50
10/10 [==============================] - 20s 2s/step - loss: 0.0280 - categorical_accuracy: 1.0000 - val_loss: 1.4066 - val_categorical_accuracy: 0.8788
Epoch 47/50
10/10 [==============================] - 20s 2s/step - loss: 0.0086 - categorical_accuracy: 1.0000 - val_loss: 1.6616 - val_categorical_accuracy: 0.8692
Epoch 48/50
10/10 [==============================] - 20s 2s/step - loss: 0.0022 - categorical_accuracy: 1.0000 - val_loss: 1.8448 - val_categorical_accuracy: 0.8596
Epoch 49/50
10/10 [==============================] - 20s 2s/step - loss: 8.3318e-04 - categorical_accuracy: 1.0000 - val_loss: 1.9540 - val_categorical_accuracy: 0.8692
Epoch 50/50
10/10 [==============================] - 20s 2s/step - loss: 5.9567e-04 - categorical_accuracy: 1.0000 - val_loss: 1.9227 - val_categorical_accuracy: 0.8692
In [9]:
dir_path=r'C:/Users/SHUBHAM/Documents/project2021/Wildlife4/New_Data_set/test1'

foriinos.listdir(dir_path):
img=image.load_img(dir_path+'//'+i,target_size=(200,200))
plt.imshow(img)
plt.show()

    x =image.img_to_array(img)
    x =np.expand_dims(x,axis=0)
    images=np.vstack([x])
val=np.argmax(model.predict(images))
ifval==0:
print("its L24")
elifval==1:
print("its L25")
elifval==2:
print("its L54")
elifval==3:
print("its L61")
elifval==4:
print("its L66")
else:
print("its L82")
