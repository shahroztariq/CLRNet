
from tensorflow.python.keras.utils import Sequence,to_categorical
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import numpy as np
import cv2
from copy import deepcopy
#   Here, `x_set` is list of path to the images,
#   `y_set` are the associated classes,
#   'v' is the no. of videos for each batch
#   and fpv is the no. of frames from each video
print('loaded')
class DFVDSequence(Sequence):
    def __init__(self, x_set, y_set, v, fpv, image_size=240,data_augmentation=True,training=True):
        self.x, self.y = x_set, y_set
        self.batch_size = fpv * v
        self.count = 0
        self.video_iterator = 0
        self.frame_iterator = 0
        self.v = v
        self.fpv = fpv
        self.dataset_size = sum([len(value) for key, value in self.x.items()])
        self.frame_counter = 0
        self.image_size = image_size
        self.data_augmentation=data_augmentation
        self.classes = []
        self.pre_classes = []
#         self.class_list=[]
        self.on_epoch_end()
        self.training=training
        

    def __len__(self):
        return int(np.ceil(self.dataset_size / float(self.batch_size)))

    def resize_img(self, img, basewidth=240):
        wpercent = (basewidth / float(img.size[0]))
        hsize = int((float(img.size[1]) * float(wpercent)))
        #         img = img.resize((basewidth,hsize), Image.ANTIALIAS)
        img = img.resize((basewidth, basewidth), Image.ANTIALIAS)
        return img

    def img_transf(self, img):
        # img -= img.mean()
        # img /= np.maximum(img.std(), 1/image_size**2) #prevent /0
        return img

    def __getitem__(self, idx):
        batch_x = []
        batch_y = []
        video_idx = 0
        _iter = 0
        while _iter < self.v:
            _iter += 1
            if self.frame_counter >= self.dataset_size:
                self.on_epoch_end()
                print('*', end=",")
                _iter -= 1
                continue
            frames2read = self.x[str(self.video_iterator)][self.frame_iterator:self.frame_iterator + self.fpv]
            temp_frames = []
            if self.data_augmentation== True:
                img_gen = ImageDataGenerator(rotation_range=30,
                                         samplewise_center=True,
                                         samplewise_std_normalization=True,
                                        # width_shift_range=0.2,
                                        # height_shift_range=0.2,
                                        # rescale=1./255,
                                        brightness_range=[0.7,1.0],
                                        channel_shift_range=50.0,
                                        # shear_range=0.1,
                                        zoom_range=0.2,
                                        horizontal_flip=True,
#                                         vertical_flip = True,
                                        fill_mode='nearest')
                transform_param = img_gen.get_random_transform(img_shape=(self.image_size, self.image_size, 3), seed=None)
            # print(transform_param)
            if len(frames2read) >= self.fpv:
                for frame in frames2read:
                    _image = cv2.imread(frame)
                    _image = cv2.cvtColor(_image, cv2.COLOR_BGR2RGB).astype('float64')
                    _image = cv2.resize(_image, (self.image_size, self.image_size))
                    if self.data_augmentation == True:
                        _image = img_gen.apply_transform(_image, transform_param)
                    _image /= 255
                    temp_frames.append(np.asarray(_image))
                batch_x.append(temp_frames)
                batch_y.append(self.y[self.video_iterator])
                self.pre_classes.append(self.y[self.video_iterator])
            else:
                _iter -= 1
            self.video_iterator += 1
            self.frame_counter += len(frames2read)
            if self.video_iterator % len(self.x) == 0:
                self.video_iterator = 0
                self.frame_iterator += len(frames2read)
            video_idx += 1
        if self.video_iterator % len(self.x) == 0:
            self.video_iterator = 0
        if len(batch_y) > 0:
            batch_y = to_categorical(batch_y, 2)
            batch_x = np.array(batch_x)
            if self.training == True:
                indices = np.arange(batch_x.shape[0])
                np.random.shuffle(indices)
                batch_x = batch_x[indices]
                batch_y = batch_y[indices]
        return (batch_x, batch_y)

    def on_epoch_end(self):
        """ Method called at the end of every epoch. """
        self.count = 0
        self.video_iterator = 0
        self.frame_iterator = 0
        self.frame_counter = 0
        self.classes=deepcopy(self.pre_classes)
        self.pre_classes=[]
        return
