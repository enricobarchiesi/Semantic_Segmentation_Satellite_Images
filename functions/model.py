import tensorflow as tf
import imageio.v2 as imageio
from sklearn.metrics import confusion_matrix


from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dropout 
from keras.layers import Conv2DTranspose
from keras.layers import concatenate
from keras.regularizers import l2

# from tensorflow.keras.layers import Input
# from tensorflow.keras.layers import Conv2D
# from tensorflow.keras.layers import MaxPooling2D
# from tensorflow.keras.layers import Dropout 
# from tensorflow.keras.layers import Conv2DTranspose
# from tensorflow.keras.layers import concatenate
# from tensorflow.keras.regularizers import l2




def conv_block(inputs=None, n_filters=32, dropout_prob=0, max_pooling=True, w_decay = 0):
    """
    Convolutional downsampling block
    
    Arguments:
        inputs -- Input tensor
        n_filters -- Number of filters for the convolutional layers
        dropout_prob -- Dropout probability
        max_pooling -- Use MaxPooling2D to reduce the spatial dimensions of the output volume
    Returns: 
        next_layer, skip_connection --  Next layer and skip connection outputs
    """

    ### START CODE HERE
    conv = Conv2D(n_filters, # Number of filters
                3,   # Kernel size   
                activation='relu',
                padding='same',
                kernel_initializer='he_normal',
                kernel_regularizer=l2(w_decay))(inputs)
    conv = Conv2D(n_filters, # Number of filters
                3,   # Kernel size   
                activation='relu',
                padding='same',
                kernel_initializer='he_normal',
                kernel_regularizer=l2(w_decay))(conv)
    ### END CODE HERE
    
    # if dropout_prob > 0 add a dropout layer, with the variable dropout_prob as parameter
    if dropout_prob > 0:
         ### START CODE HERE
        conv = Dropout(dropout_prob)(conv)
         ### END CODE HERE
         
        
    # if max_pooling is True add a MaxPooling2D with 2x2 pool_size
    if max_pooling:
        ### START CODE HERE
        next_layer = MaxPooling2D(2,strides=2)(conv)
        ### END CODE HERE
        
    else:
        next_layer = conv
        
    skip_connection = conv
    
    return next_layer, skip_connection


def upsampling_block(expansive_input, contractive_input, n_filters=32):
    """
    Convolutional upsampling block
    
    Arguments:
        expansive_input -- Input tensor from previous layer
        contractive_input -- Input tensor from previous skip layer
        n_filters -- Number of filters for the convolutional layers
    Returns: 
        conv -- Tensor output
    """
    
    ### START CODE HERE
    up = Conv2DTranspose(
                 n_filters,    # number of filters
                 3,    # Kernel size
                 strides=2,
                 padding='same')(expansive_input)
    
    # Merge the previous output and the contractive_input
    merge = concatenate([up, contractive_input], axis=3)
    conv = Conv2D(n_filters,   # Number of filters
                 3,     # Kernel size
                 activation='relu',
                 padding='same',
                 kernel_initializer='he_normal')(merge)
    conv = Conv2D(n_filters,   # Number of filters
                 3,     # Kernel size
                 activation='relu',
                 padding='same',
                 kernel_initializer='he_normal')(conv)
    ### END CODE HERE
    
    return conv



def unet_model(n_classes, 
               tile_width, 
               tile_height, 
               num_bands, 
               n_filters=32, 
               w_decay = 1e-5, 
               droprate = 0.3):
    """
    Unet model
    
    Arguments:
        input_size -- Input shape 
        n_filters -- Number of filters for the convolutional layers
        n_classes -- Number of output classes
    Returns: 
        model -- tf.keras.Model
    """
    input_size = (tile_width, tile_height, num_bands)
    inputs = Input(input_size)
    # Contracting Path (encoding)
    # Add a conv_block with the inputs of the unet_ model and n_filters
    ### START CODE HERE
    cblock1 = conv_block(inputs=inputs, n_filters=n_filters*1)
    # Chain the first element of the output of each block to be the input of the next conv_block. 
    # Double the number of filters at each new step
    cblock2 = conv_block(inputs=cblock1[0], n_filters=n_filters*2, w_decay=w_decay)
    cblock3 = conv_block(inputs=cblock2[0], n_filters=n_filters*4, w_decay=w_decay, dropout_prob=droprate*0.5)
    cblock4 = conv_block(inputs=cblock3[0], n_filters=n_filters*8,w_decay=w_decay, dropout_prob=droprate) # Include a dropout_prob of 0.3 for this layer
    # Include a dropout_prob of 0.3 for this layer, and avoid the max_pooling layer
    cblock5 = conv_block(inputs=cblock4[0], n_filters=n_filters*16, w_decay=w_decay, dropout_prob=droprate*1.5, max_pooling=False)
    ### END CODE HERE
    
    # Expanding Path (decoding)
    # Add the first upsampling_block.
    # Use the cblock5[0] as expansive_input and cblock4[1] as contractive_input and n_filters * 8
    ### START CODE HERE
    ublock6 = upsampling_block(cblock5[0], cblock4[1], n_filters*8)
    # Chain the output of the previous block as expansive_input and the corresponding contractive block output.
    # Note that you must use the second element of the contractive block i.e before the maxpooling layer. 
    # At each step, use half the number of filters of the previous block 
    ublock7 = upsampling_block(ublock6, cblock3[1], n_filters*4)
    ublock8 = upsampling_block(ublock7, cblock2[1], n_filters*2)
    ublock9 = upsampling_block(ublock8, cblock1[1], n_filters*1)
    ### END CODE HERE

    conv9 = Conv2D(n_filters,
                3,
                activation='relu',
                padding='same',
                kernel_initializer='he_normal')(ublock9)

    # Add a Conv2D layer with n_classes filter, kernel size of 1 and a 'same' padding
    ### START CODE HERE
    conv10 = Conv2D(n_classes, 1, activation='softmax', padding='same')(conv9)
    ### END CODE HERE
    
    model = tf.keras.Model(inputs=inputs, outputs=conv10)

    return model




def unet_model_modular(n_classes, 
                       tile_width, 
                       tile_height, 
                       num_bands, 
                       n_blocks, 
                       n_filters=32, 
                       w_decay=1e-5, 
                       droprate=0.3,
                       drop_multiplier=None, 
                       weight_multiplier=None, 
                       filter_growth = 2):
    
    input_size = (tile_width, tile_height, num_bands)
    inputs = Input(input_size)
    x = inputs  # Initialize x to be the input tensor

    contracting_blocks = []  # List to store the contractive blocks
    

    if drop_multiplier is None:
        drop_multiplier = [1.0] * n_blocks  # Default: no modification to weight decay rates
    
    if weight_multiplier is None:
        weight_multiplier = [1.0] * n_blocks  # Default: no modification to dropout rates

    for i in range(n_blocks):
        if i < n_blocks-1:
            x, skip = conv_block(inputs=x, n_filters=n_filters, w_decay=w_decay* weight_multiplier[i], dropout_prob=droprate * drop_multiplier[i])
            contracting_blocks.append(skip)
        else:
            x, skip = conv_block(inputs=x, n_filters=n_filters, w_decay=w_decay* weight_multiplier[i], dropout_prob=droprate * drop_multiplier[i], max_pooling=False)
        n_filters *= filter_growth  # Double the number of filters in each downsampling block
        
    contracting_blocks.reverse()
    n_filters //= filter_growth
    for i in range(n_blocks-1):
        n_filters //= filter_growth  # Halve the number of filters in each upsampling block
        x = upsampling_block(x, contracting_blocks[i], n_filters=n_filters)
        # x = upsampling_block(x, contracting_blocks.pop(), n_filters=n_filters)
        
        
    output_1 = Conv2D(n_filters,
            3,
            activation='relu',
            padding='same',
            kernel_initializer='he_normal')(x)

    output = Conv2D(n_classes, 1, activation='softmax', padding='same')(output_1)

    model = tf.keras.Model(inputs=inputs, outputs=output)

    return model