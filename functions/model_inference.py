import numpy as np
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()
from functions.gis import *
from functions.smooth_tiled_predictions import predict_img_with_smooth_windowing


def image_inference(input_img_pth, output_img_pth, model_dict, tile_size, n_classes, EPSG_code=2169, argmax = True):
    model = model_dict['model_info']['model']
    batch_size = model_dict['model_info']['params']['batch_size']
    input_img = gdal.Open(input_img_pth).ReadAsArray().transpose(1,2,0)
    # input_img = np.expand_dims(input_img, axis=0)

    # print(input_img.shape)


    # Use the algorithm. The `pred_func` is passed and will process all the image 8-fold by tiling small patches with overlap, called once with all those image as a batch outer dimension.
    # Note that model.predict(...) accepts a 4D tensor of shape (batch, x, y, nb_channels), such as a Keras model.
    predictions_smooth = predict_img_with_smooth_windowing(
        input_img,
        model,
        window_size=tile_size,
        subdivisions=2,  # Minimal amount of overlap for windowing. Must be an even number.
        nb_classes=n_classes,
        batch_size=batch_size
    )
    if argmax:
        argmax_result = np.argmax(predictions_smooth, axis=2)
        upload_tif(out_file_pth=output_img_pth, arr=argmax_result, old_file_pth=input_img_pth, EPSG_code = EPSG_code)
    else:
        return predictions_smooth
    
