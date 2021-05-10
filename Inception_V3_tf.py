# We will build the architecture of Inception V3 using keras API from tensorflow
from tensorflow import keras
from keras.layers import Conv2D,AveragePooling2D,Dropout,Flatten,concatenate,MaxPool2D,Input,Dense
from keras.models import Model

# We need to create separate function for inception block
def create_inception_block(x,filters_1x1,filters_3x3_reduce,filters_3x3,filters_5x5_reduce,filters_5x5,filters_pool_proj,name):
    #default kernel initializer is used in this architecture glorot_uniform
    #default bias initializer is used 'zeros'
    conv_1x1 = Conv2D(filters=filters_1x1,kernel_size=(1,1),strides=(1,1),padding='same',activation='relu')(x)
    conv_3x3 = Conv2D(filters=filters_3x3_reduce,kernel_size=(1,1),strides=(1,1),padding='same',activation='relu')(x)
    conv_3x3 = Conv2D(filters=filters_3x3,kernel_size=(3,3),strides=(1,1),padding='same',activation='relu')(conv_3x3)
    conv_5x5 = Conv2D(filters=filters_5x5_reduce,kernel_size=(1,1),strides=(1,1),padding='same',activation='relu')(x)
    conv_5x5 = Conv2D(filters=filters_5x5,kernel_size=(5,5),strides=(1,1),padding='same',activation='relu')(conv_5x5)
    max_pool = MaxPool2D(pool_size=(3,3),strides=(1,1),padding='same')(x)
    pool_proj = Conv2D(filters=filters_pool_proj,kernel_size=(1,1),strides=(1,1),padding='same',activation='relu')(max_pool)

    #Concatenation of channels from the inception block 
    output = concatenate([conv_1x1,conv_3x3,conv_5x5,pool_proj],axis=3,name=name)

    return output

def inception_model():
    inputs = Input(shape=(224,224,3),name='input_layer')
    x = Conv2D(filters = 64, kernel_size=(7,7),strides=(2,2),padding='same',activation='relu',name='conv_layer_1')(inputs)
    x = MaxPool2D(pool_size=(3,3),strides=(2,2),padding='same',name='max_pool_layer_1')(x)
    x = Conv2D(filters = 192, kernel_size=(3,3),strides=(1,1),padding='same',activation='relu',name='conv_layer_2')(x)
    x = MaxPool2D(pool_size=(3,3),strides=(2,2),padding='same',name='max_pool_layer_2')(x)
    # params will be in the order of x(previous layer) , filters_1x1,filters_3x3_reduce,filters_3x3,
    # filters_5x5_reduce,filters_5x5,filters_pool_proj,name
    x = create_inception_block(x,64,96,128,16,32,32,name='inception_3a')
    x = create_inception_block(x,128,128,192,32,96,64,name='inception_3b')
    x = MaxPool2D(pool_size=(3,3),strides=(2,2),padding='same',name='max_pool_layer_3')(x)
    x = create_inception_block(x,192,96,208,16,48,64,name='inception_4a')
    x = create_inception_block(x,160,112,224,24,64,64,name='inception_4b')
    x = create_inception_block(x,128,128,256,24,64,64,name='inception_4c')
    x = create_inception_block(x,112,144,288,32,64,64,name='inception_4d')
    x = create_inception_block(x,256,160,320,32,128,128,name='inception_4e')
    x = MaxPool2D(pool_size=(3,3),strides=(2,2),padding='same',name='max_pool_layer_4')(x)
    x = create_inception_block(x,256,160,320,32,128,128,name='inception_5a')
    x = create_inception_block(x,384,192,384,48,128,128,name='inception_5b')
    x = AveragePooling2D(pool_size=(7,7),strides=(1,1),name='average_pool_1')(x)
    x = Dropout(0.4)(x)
    x = Flatten()(x)
    x = Dense(1000,activation='linear',name='FC1')(x)
    output_3 = Dense(1000,activation='softmax',name='softmax')(x)
    model_1 = Model(inputs,output_3)
    return model_1

if __name__ == '__main__':
    model_inception = inception_model()
    print(model_inception.summary())
    keras.utils.plot_model(model_inception,to_file='inception_v3.png',show_shapes=True)