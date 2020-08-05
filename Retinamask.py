import keras
import keras_retinanet.losses
#use the below command when you are giving your own config file
#from keras_retinanet.utils.config import read_config_file, parse_anchor_parameters
from keras_retinanet.utils.gpu import setup_gpu
from keras_retinanet.utils.model import freeze as freeze_model
from keras_retinanet.utils.transform import random_transform_generator
from keras_maskrcnn import losses
from keras_maskrcnn import models
from keras_maskrcnn.utils.visualization import draw_mask
from keras_retinanet.utils.visualization import draw_box, draw_caption, draw_annotations
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.colors import label_color
import matplotlib.pyplot as plt
import cv2
import numpy as np
import time
setup_gpu(0)

class Retinamask :
    
    def __init__(self,annotations,classes,Input_weights_path,trained_weights_path,test_image_path,output_image_path,epoch):
        self.annotations=annotations
        self.classes=classes
        self.Input_weights_path=Input_weights_path
        self.trained_weights_path=trained_weights_path
        self.test_image_path=test_image_path
        self.output_image_path=output_image_path
        self.epoch=epoch
    
    def create_models(backbone_retinanet, num_classes, weights, freeze_backbone=False, class_specific_filter=True, anchor_params=None):
        def model_with_weights(model, weights, skip_mismatch):
            if weights is not None:
                model.load_weights(weights, by_name=True, skip_mismatch=skip_mismatch)
            return model

        modifier = freeze_model if freeze_backbone else None
        model= model_with_weights(backbone_retinanet(num_classes,nms=True,class_specific_filter=class_specific_filter,modifier=modifier,anchor_params=anchor_params), weights=weights, skip_mismatch=True)
        training_model   = model
        prediction_model = model

        # compile model
        training_model.compile(loss={'regression'    : keras_retinanet.losses.smooth_l1(),'classification': keras_retinanet.losses.focal(),
            'masks': losses.mask(),},optimizer=keras.optimizers.adam(lr=1e-5, clipnorm=0.001))
        return model, training_model, prediction_model

    def create_generators(batch_size,annotations,classes):
        # create random transform generator for augmenting training data
        transform_generator = random_transform_generator(flip_x_chance=0.5)
        from keras_maskrcnn.preprocessing.csv_generator import CSVGenerator
        train_generator = CSVGenerator(annotations,classes)
        validation_generator = None    
        return train_generator, validation_generator


class Retinamask1(Retinamask):     
    
    def retinamask(self):
        backbone = models.backbone('resnet50')
        batch_size=1
        train_generator, validation_generator = Retinamask.create_generators(batch_size,self.annotations,self.classes)
        freeze_backbone='store_true'
        weights =self.Input_weights_path
        print('Creating model, this may take a second...')
        model, training_model, prediction_model =Retinamask.create_models(backbone_retinanet=backbone.maskrcnn,num_classes=train_generator.num_classes(),weights=weights,freeze_backbone=freeze_backbone)
        #print(model.summary())
        training_model.fit_generator(generator=train_generator,steps_per_epoch=1000,epochs=self.epoch,verbose=1,max_queue_size=1)
        training_model.save(self.trained_weights_path+'retinamask.h5')
     
        #Testing
        model_path=self.trained_weights_path+'retinamask.h5'
        model = models.load_model(model_path, backbone_name='resnet50')
        labels_to_names={0:'ship'}
        # load image
        image = read_image_bgr(test_image_path)
        # copy to draw on
        draw = image.copy()
        draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
        # preprocess image for network
        image = preprocess_image(image)
        image, scale = resize_image(image)
        # process image
        start = time.time()
        outputs = model.predict_on_batch(np.expand_dims(image, axis=0))
        print("processing time: ", time.time() - start)
        boxes  = outputs[-4][0]
        scores = outputs[-3][0]
        labels = outputs[-2][0]
        masks  = outputs[-1][0]
        # correct for image scale
        boxes /= scale
        # visualize detections
        for box, score, label, mask in zip(boxes, scores, labels, masks):
            if score < 0.5:
                break
            color = label_color(label)
            b = box.astype(int)
            draw_box(draw, b, color=color)
            mask = mask[:, :, label]
            draw_mask(draw, b, mask, color=label_color(label))
            caption = "{} {:.3f}".format(labels_to_names[label], score)
            draw_caption(draw, b, caption)
            plt.imsave(self.output_image_path+'output.jpg',draw)

if __name__ == "__main__":
    annotations=input('Enter the path of annotations file:')
    classes=input('Enter the path of classes file:')
    Input_weights_path=input('Enter the Input coco weights path:')
    trained_weights_path=input('Enter the path to save trained weights path:')
    test_image_path=input('Enter the path of test input images:')
    output_image_path=input('Enter the path of  output_image:')
    epoch=int(input('Enter how many epochs you want to run:'))
    a=Retinamask1(annotations,classes,Input_weights_path,trained_weights_path,test_image_path,output_image_path,epoch)
    b=a.retinamask()

