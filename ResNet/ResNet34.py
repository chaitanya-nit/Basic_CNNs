import numpy as np
from tensorflow.keras.layers import Dense,Conv2D,GlobalAveragePooling2D,MaxPool2D,Input,BatchNormalization,ReLU,Add,Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model


# Let us define a residual block which we can use it in our model
def residual_block(input_layer,block_size,filters,filter_size,name,padding='same'):
    x = input_layer
    short_cut_layer = Conv2D(filters=filters,kernel_size=(1,1),strides=(1,1),padding=padding,name=name)(x)
    short_cut_layer = BatchNormalization()(short_cut_layer)
    for i in range(block_size):
        x = Conv2D(filters=filters,kernel_size=filter_size,padding=padding)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2D(filters=filters,kernel_size=filter_size,padding=padding)(x)
        x = BatchNormalization()(x)
        x = Add()([short_cut_layer,x])
        x = ReLU()(x)
        short_cut_layer = x
    return x

def create_residual_model():
    inputs = Input(shape=(224,224,3),batch_size=32)
    conv1 = Conv2D(filters=64,kernel_size=(7,7),strides=(2,2),padding='same',name='conv1')(inputs)
    batch_norm_1 = BatchNormalization()(conv1)
    activation_1 = ReLU()(batch_norm_1)
    max_pool_1 = MaxPool2D(pool_size=(3,3),strides=(2,2),padding='same',name='max_pool1')(activation_1)
    residual_block_1 = residual_block(max_pool_1,3,64,(3,3),'residual_block_1','same')
    residual_block_2 = residual_block(residual_block_1,4,128,(3,3),'residual_block_size_2',padding='same')
    residual_block_3 = residual_block(residual_block_2,6,256,(3,3),'residual_block_size_3',padding='same')
    residual_block_4 = residual_block(residual_block_3,3,512,(3,3),'residual_block_size_4',padding='same')
    avg_pool = GlobalAveragePooling2D()(residual_block_4)
    flatten_layer = Flatten()(avg_pool)
    outputs = Dense(1000,activation='softmax')(flatten_layer)
    return Model(inputs,outputs)

restnet34 = create_residual_model()
print(restnet34.summary())
plot_model(restnet34,to_file='sample_resnet.png',show_shapes=True)
random_image = np.random.random(size=(1,224,224,3))
output = restnet34(random_image)
print(output.shape)
restnet34.save('resnet34.h5')