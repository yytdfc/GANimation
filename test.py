import os
import argparse
import glob
import cv2
from utils import face_utils
from utils import cv_utils
import face_recognition
from PIL import Image
import torchvision.transforms as transforms
import torch
import pickle
import numpy as np
from models.models import ModelsFactory
from options.test_options import TestOptions
import copy

AU_LIST = [1, 2, 4, 5, 6, 7, 9, 10, 12, 14, 15, 17, 20, 23, 25, 26, 45]
EMOTION = {
    'Happiness': [6, 12],
    'Sadness': [1,4,15],
    'Surprise': [1,2,5,26],
    'Fear': [1,2,4,5,7,20,26],
    'Anger': [4,5,7,23],
    'Disgust': [9,15],
    'Contempt': [12, 14],
}

# au -> emotion 'From Emotions to Action Units with Hidden and Semi-Hidden-Task Learning'
EMOTIONS = ['Anger', 'Disgust', 'Fear', 'Happyness', 'Sadness', 'Surprise', 'Neutral']
AU_IDX = [1, 2, 4, 5, 6, 7, 9, 10, 12, 15, 17, 20, 25, 26]
AU_TABLE = [
    [0.17,  0.1 ,  0.33,  0.25,  0.03,  0.05,  0.  ,  0.1 ,  0.  , 0.05,  0.06,  0.4 ,  0.31,  0.49],
    [0.01,  0.01,  0.35,  0.01,  0.06,  0.36,  0.06,  0.21,  0.  , 0.  ,  0.  ,  0.25,  0.32,  0.4 ],
    [0.12,  0.01,  0.33,  0.55,  0.  ,  0.29,  0.  ,  0.03,  0.  , 0.04,  0.04,  0.25,  0.2 ,  0.75],
    [0.07,  0.09,  0.01,  0.05,  0.94,  0.01,  0.05,  0.  ,  0.92, 0.  ,  0.  ,  0.02,  0.34,  0.55],
    [0.22,  0.01,  0.25,  0.  ,  0.03,  0.39,  0.  ,  0.05,  0.05, 0.09,  0.17,  0.07,  0.14,  0.2 ],
    [0.15,  0.19,  0.08,  0.76,  0.  ,  0.02,  0.  ,  0.1 ,  0.04, 0.  ,  0.04,  0.09,  0.26,  0.72],
    [0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  , 0.  ,  0.  ,  0.  ,  0.  ,  0.  ]
]

class MorphFacesInTheWild:
    def __init__(self, opt):
        self._opt = opt
        self._model = ModelsFactory.get_by_name(self._opt.model, self._opt)
        self._model.set_eval()
        self._transform = transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                                   std=[0.5, 0.5, 0.5])
                                              ])

    def morph_file(self, img_path, expresion, output_surfix='_out.png'):
        img = cv_utils.read_cv2_img(img_path)
        morphed_img = self._img_morph(img, expresion)
        # output_name = '%s_%s' % (os.path.basename(img_path), output_surfix)
        # self._save_img(morphed_img, output_name)
        return copy.copy(morphed_img)

    def _img_morph(self, img, expresion):
        bbs = face_recognition.face_locations(img)
        if len(bbs) > 0:
            y, right, bottom, x = bbs[0]
            bb = x, y, (right - x), (bottom - y)
            face = face_utils.crop_face_with_bb(img, bb)
            face = face_utils.resize_face(face)
        else:
            face = face_utils.resize_face(img)

        morphed_face = self._morph_face(face, expresion)

        return morphed_face

    def _morph_face(self, face, expresion):
        face = torch.unsqueeze(self._transform(Image.fromarray(face)), 0)
        expresion = torch.unsqueeze(torch.from_numpy(expresion/1.0), 0)
        test_batch = {'real_img': face, 'real_cond': expresion, 'desired_cond': expresion, 'sample_id': torch.FloatTensor(), 'real_img_path': []}
        self._model.set_input(test_batch)
        imgs, _ = self._model.forward(keep_data_for_visuals=False, return_estimates=True)
        return imgs['real_img'], imgs['fake_imgs_masked'], imgs['fake_imgs']

    def _save_img(self, img, filename):
        filepath = os.path.join(self._opt.output_dir, filename)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filepath, img)


def main0():
    opt = TestOptions().parse()
    if not os.path.isdir(opt.output_dir):
        os.makedirs(opt.output_dir)

    morph = MorphFacesInTheWild(opt)

    image_path = opt.input_path
    expression = np.random.uniform(0, 1, opt.cond_nc)
    morph.morph_file(image_path, expression)


def main():
    au_dict = {}
    for i, a in enumerate(AU_LIST):
        au_dict[a] = i
    print(au_dict)
    emotions_au = []
    for au in AU_TABLE:
        expression = np.zeros(len(AU_LIST))
        for i, a in zip(AU_IDX, au):
            expression[au_dict[i]] = a
        emotions_au.append( expression)
    print(emotions_au)

    opt = TestOptions().parse()
    if not os.path.isdir(opt.output_dir):
        os.makedirs(opt.output_dir)

    morph = MorphFacesInTheWild(opt)

    with open(opt.input_path, 'r') as f:
        big_img = []
        for image_path in f.readlines()[:10]:
            image_path = opt.data_dir + '/' + opt.images_folder + '/' + image_path.strip()
            print(image_path)
            imgs = []
            for expression in emotions_au:
                real_img, img, mask = morph.morph_file(image_path, expression)
                imgs.append(np.concatenate([img,
                                            mask
                                            ]))
            imgs = [np.concatenate([np.array(real_img),
                                    np.zeros((128, 128, 3), dtype=np.uint8) + 255
                                    ])] + imgs
            imgs = np.concatenate(imgs, 1)
            big_img.append(imgs)
            # morph._save_img(imgs, image_path.split('/')[-1])
            # break
        big_img = np.concatenate(big_img, 0)
        morph._save_img(big_img, 'test_epoch%d.png' % (opt.load_epoch))



if __name__ == '__main__':
    main()
