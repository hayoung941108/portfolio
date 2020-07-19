import csv
import pickle
import urllib
from urllib.request import urlopen
from bs4 import BeautifulSoup as bs
import requests
import imageio
import json
import math
import warnings

from project.class01_Face_Distance_Ratio import FaceDistanceRatio

warnings.filterwarnings(action='ignore')
import os
import cv2
import glob
import dlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.transform import pyramid_reduce
from skimage.transform import pyramid_expand
from keras.layers import Conv2D, Input, Activation
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from project.keras_01_Subpixel import Subpixel
from project.keras_02_Datagenerator import DataGenerator


class CrawlingRich:
    # base_path = r"C:\labs\\project\\project\\celeba-dataset"
    def __init__(self, base_path):
        self.base_path = base_path

    def img_resolution_train(self):
        img_base_path = os.path.join(self.base_path, "img_align_celeba")
        target_img_path = os.path.join(self.base_path, "processed")

        eval_list = np.loadtxt(os.path.join(self.base_path, "list_eval_partition.csv"),
                               dtype=str, delimiter=',', skiprows=1)

        img_sample = cv2.imread(os.path.join(img_base_path, eval_list[0][0]))

        h, w, _ = img_sample.shape
        print(h, w)
        # 이미지로 crop
        crop_sample = img_sample[int((h - w) / 2):int(-(h - w) / 2), :]

        # 이미지 4배 축소 후 normalize
        resized_sample = pyramid_reduce(crop_sample, downscale=4, multichannel=True)  # 컬러채널 허용

        pad = int((crop_sample.shape[0] - resized_sample.shape[0]) / 2)

        padded_sample = cv2.copyMakeBorder(resized_sample, top=pad + 1, bottom=pad,
                                           left=pad + 1, right=pad, borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))

        print(crop_sample.shape, padded_sample.shape)

        # fig, ax = plt.subplots(1, 4, figsize=(12, 5))
        # ax = ax.ravel()
        # ax[0].imshow(img_sample)
        # ax[1].imshow(crop_sample)
        # ax[2].imshow(resized_sample)
        # ax[2].imshow(cv2.resize(resized_sample, dsize=(45, 45)))
        # ax[3].imshow(padded_sample)
        # plt.show()

        downscale = 4

        # 이미지 train 할 파일 생성
        for i, e in enumerate(eval_list):
            if i == (len(eval_list) - 1):
                break
            filename, ext = os.path.splitext(e[0])
            img_path = os.path.join(img_base_path, e[0])
            img = cv2.imread(img_path)
            h, w, _ = img.shape
            crop = img[int((h - w) / 2):int(-(h - w) / 2), :]
            crop = cv2.resize(crop, dsize=(176, 176))
            resized = pyramid_reduce(crop, downscale=downscale, multichannel=True)  # 컬러 채널 허용
            norm = cv2.normalize(crop.astype(np.float64), None, 0, 1, cv2.NORM_MINMAX)

            if int(e[1]) == 0:  # Train
                np.save(os.path.join(target_img_path, "x_train", filename + ".npy"), resized)
                np.save(os.path.join(target_img_path, "y_train", filename + ".npy"), norm)
            else:  # Validation
                np.save(os.path.join(target_img_path, "x_val", filename + ".npy"), resized)
                np.save(os.path.join(target_img_path, "y_val", filename + ".npy"), norm)

        return None

    def img_resolution_train2(self):
        # train, validation, test 파일 list 만들기
        x_train_list = sorted(glob.glob(os.path.join(self.base_path, 'processed\\x_train', '*.npy')))
        x_val_list = sorted(glob.glob(os.path.join(self.base_path, 'processed\\x_val', '*.npy')))

        print(len(x_train_list), len(x_val_list))
        print(x_train_list[0])


        x1 = np.load(x_train_list[0])
        x2 = np.load(x_val_list[0])

        print(x1.shape, x2.shape)

        # plt.subplot(1, 2, 1)
        # plt.imshow(x1)
        # plt.subplot(1, 2, 2)
        # plt.imshow(x2)
        # plt.show()

        train_gen = DataGenerator(list_IDs=x_train_list, labels=None, batch_size=16, dim=(44,44),
                                  n_channels=3, n_classes=None, shuffle=True)

        val_gen = DataGenerator(list_IDs=x_val_list, labels=None, batch_size=16, dim=(44,44),
                                n_channels=3, n_classes=None, shuffle=False)

        upscale_factor = 4
        inputs = Input(shape=(44, 44, 3))

        net = Conv2D(filters=64, kernel_size=5, strides=1, padding='same', activation='relu')(inputs)
        net = Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu')(net)
        net = Conv2D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu')(net)
        net = Conv2D(filters=upscale_factor**2, kernel_size=3, strides=1, padding='same', activation='relu')(net)
        net = Subpixel(filters=3, kernel_size=3, r=upscale_factor, padding='same')(net)

        outputs = Activation('relu')(net)

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss='mse')
        # model.save('model.h5')
        model.save(os.path.join(self.base_path, 'model.h5'))
        model.summary()

        history = model.fit_generator(train_gen, validation_data=val_gen, epochs=10, verbose=1, callbacks=[
            ModelCheckpoint(r"C:\labs\\project\\project\\model.h5",  # 풀 주소로 적어야 에러가 안 생김
                            monitor='val_loss', verbose=1, save_best_only=True)])
            # ModelCheckpoint(os.path.join(self.base_path, 'model.h5'),  # 풀 주소로 적어야 에러가 안 생김
            #                 monitor='val_loss', verbose=1, save_best_only=True)])

        return history

    def sum1forline(self, filename):
        self.filename = filename
        with open(filename) as f:
            return sum(1 for line in f)

    def img_resolution_test(self):
        img_base_path = os.path.join(self.base_path, "imageset_test")
        target_img_path = os.path.join(self.base_path, "processed_test")

        eval_list = np.loadtxt(os.path.join(self.base_path, "list_eval_partition_test.csv"),
                               dtype=str, delimiter=',', skiprows=1)

        img_sample = cv2.imread(os.path.join(img_base_path, eval_list[0][0]))

        h, w, _ = img_sample.shape
        print(h, w)

        # 정사각형 이미지로 crop
        crop_sample = img_sample[int((h - w) / 2):int(-(h - w) / 2), :]

        # 이미지 4배 축소 후 normalize
        resized_sample = pyramid_reduce(crop_sample, downscale=4, multichannel=True)  # 컬러채널 허용

        pad = int((crop_sample.shape[0] - resized_sample.shape[0]) / 2)

        padded_sample = cv2.copyMakeBorder(resized_sample, top=pad + 1, bottom=pad,
                                           left=pad + 1, right=pad, borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))

        print(crop_sample.shape, padded_sample.shape)

        # fig, ax = plt.subplots(1, 4, figsize=(12, 5))
        # ax = ax.ravel()
        # ax[0].imshow(img_sample)
        # ax[1].imshow(crop_sample)
        # ax[2].imshow(resized_sample)
        # ax[2].imshow(cv2.resize(resized_sample, dsize=(45, 45)))
        # ax[3].imshow(padded_sample)
        # plt.show()

        downscale = 4

        # 이미지 test 할 파일 생성
        for i, e in enumerate(eval_list):
            if i == self.sum1forline(os.path.join(self.base_path, "list_eval_partition_test.csv")):
                break
            filename, ext = os.path.splitext(e[0])
            img_path = os.path.join(img_base_path, e[0])
            img = cv2.imread(img_path)
            h, w, _ = img.shape
            crop = img[int((h - w) / 2):int(-(h - w) / 2), :]
            crop = cv2.resize(crop, dsize=(176, 176))
            resized = pyramid_reduce(crop, downscale=downscale, multichannel=True)  # 컬러 채널 허용
            norm = cv2.normalize(crop.astype(np.float64), None, 0, 1, cv2.NORM_MINMAX)

            if int(e[1]) == 2:  # Test
                np.save(os.path.join(target_img_path, "x_test", filename + ".npy"), resized)
                np.save(os.path.join(target_img_path, "y_test", filename + ".npy"), norm)

        return None

    def img_resolution_test2(self):
        x_test_list = sorted(glob.glob(os.path.join(self.base_path, 'processed_test/x_test', '*.npy')))
        y_test_list = sorted(glob.glob(os.path.join(self.base_path, 'processed_test/y_test', '*.npy')))
        print(len(x_test_list), len(y_test_list))
        # print(x_test_list[0])

        test_idx = -1

        # 저해상도 이미지(input)
        x1_test = np.load(x_test_list[test_idx])

        # 저해상도 이미지 확대시킨 이미지
        x1_test_resized = pyramid_expand(x1_test, 4, multichannel=True)  # 색깔 채널 조건 추가

        # 정답 이미지
        y1_test = np.load(y_test_list[test_idx])

        # 모델 만들기
        upscale_factor = 4
        inputs = Input(shape=(44, 44, 3))

        net = Conv2D(filters=64, kernel_size=5, strides=1, padding='same', activation='relu')(inputs)
        net = Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu')(net)
        net = Conv2D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu')(net)
        net = Conv2D(filters=upscale_factor ** 2, kernel_size=3, strides=1, padding='same', activation='relu')(net)
        net = Subpixel(filters=3, kernel_size=3, r=upscale_factor, padding='same')(net)

        outputs = Activation('relu')(net)

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss='mse')
        # model.save('model.h5')

        # 모델이 예측한 이미지(output)
        y_pred = model.predict(x1_test.reshape((1, 44, 44, 3)))
        print(x1_test.shape, y1_test.shape)

        x1_test = (x1_test * 255).astype(np.uint8)
        x1_test_resized = (x1_test_resized * 255).astype(np.uint8)
        y1_test = (y1_test * 255).astype(np.uint8)
        y_pred = np.clip(y_pred.reshape((176, 176, 3)), 0, 1)

        x1_test = cv2.cvtColor(x1_test, cv2.COLOR_BGR2RGB)
        x1_test_resized = cv2.cvtColor(x1_test_resized, cv2.COLOR_BGR2RGB)
        y1_test = cv2.cvtColor(y1_test, cv2.COLOR_BGR2RGB)
        y_pred = cv2.cvtColor(y_pred, cv2.COLOR_BGR2RGB)

        # fig, ax = plt.subplots(1,4,figsize=(15, 10))
        # ax = ax.ravel()
        #
        # ax[0].set_title('input')
        # ax[0].imshow(x1_test)
        #
        # ax[1].set_title('resized')
        # ax[1].imshow(x1_test_resized)
        #
        # ax[2].set_title('output')
        # ax[2].imshow(y_pred)
        #
        # ax[3].set_title('groundtruth')
        # ax[3].imshow(y1_test)
        #
        # plt.show()

        return None

    def del_csv(self):
        df = pd.read_csv(os.path.join(self.base_path, 'list_eval_partition_test.csv'))
        df = df.drop([df.index[-1]])
        df.to_csv(os.path.join(self.base_path, 'list_eval_partition_test.csv'), index=False)

    def Distance(self, a, b):
        self.a = a
        self.b = b
        x1, y1, x2, y2 = a[0], a[1], b[0], b[1]
        distance = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        return distance

    def Radian(self, a, b):
        self.a = a
        self.b = b
        x1, y1, x2, y2 = a[0], a[1], b[0], b[1]
        c = [x1, y2]
        A = self.Distance(a, b)
        # B = Distance(b, c)
        C = self.Distance(a, c)
        angle = math.acos(C / A)
        dgrees = math.degrees(angle)
        return int(dgrees)

    def facing_67(self, face_folder_path, rank):
        self.face_folder_path = face_folder_path
        self.rank = rank
        predictor_path = os.path.join(self.base_path, 'shape-predict/shape_predictor_68_face_landmarks.dat')
        # predictor_path = "../testing/shape-predict/shape_predictor_68_face_landmarks.dat"
        faces_folder_path = face_folder_path

        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(predictor_path)

        n = 1  # 이미지 번호 증가
        cnt = 0  # 제대로 저장된 이미지 수
        no_dets = 0  # 데이터를 찾지 못한 이미지 수
        minus_value = 0  # 데이터는 있지만 음수가 들어가 있는 이미지 수
        remove_counts = 0

        ALL = list(range(0, 68))

        # dict를 저장하는 리스트
        landmark_dict_list = []

        for f in glob.glob(os.path.join(faces_folder_path, f"{rank}*.jpg")):
            img = dlib.load_rgb_image(f)  # directory 에 있는 이미지 파일 하나씩 불러오기
            r, g, b = cv2.split(img)  # bgr을 rgb 로변경
            cvImg = cv2.merge([b, g, r])
            cvImg = cv2.bilateralFilter(cvImg, 9, 75, 75)

            dets = detector(cvImg, 1)  # face detecting function. 이 결과로 rectangle[] value return
            if len(dets) == 1:  # rectangle[] 에 값이 하나인 경우.

                minus_detector = int(str(dets[0]).replace('[', '').replace(']', '').replace('(', '').replace(')', ''). \
                                     replace(',', '').replace(' ', '').replace('-',
                                                                               '1111111'))  # rectangle() value에 음수가 있는 경우 거르기 위한 과정

                # minus_detector 를 만든 이유는 위에서 dets 가 있는 경우를 걸러도 rectangle() value에 음수가 되는 값이 존재할 수 있기 때문이다.
                # 만일 rectangle() value가 음수가 되면 47번째 'crop = ~' 라인에서 Assertion: _!is_empty() 에러를 반환하는 경우가 발생한다.
                # 구글에서 90개의 이미지를 크롤링한 결과 1장의 rectangle() value가 음수가 되었고 이를 거르는 과정을 설정하였다.

                if minus_detector < 100_000_000_000:  # 위 과정에서 음수가 최소 12자리 되도록 설정 -> xx가 양수이면 1000억 미만
                    for _, d in enumerate(dets):
                        shape = predictor(img, d)

                        # 좌표를 저장할 빈 리스트 생성
                        landmark_list = []
                        for i in range(0, shape.num_parts):
                            x = shape.part(i).x
                            y = shape.part(i).y

                            # 좌표값 landmark_list에 저장
                            landmark_list.append([x, y])

                            cv2.putText(cvImg, str(i), (x, y), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.3, (0, 255, 0))

                        if self.Radian(landmark_list[27], landmark_list[30]) < 9:
                            # crop = cvImg[d.top():d.bottom(), d.left():d.right()]
                            # cv2.imwrite(f'rich{n}.jpg', crop)  ###@ 여기서 왜 저장 하지????? ### 그리고 이거 실행하면 5개 밖에 사진이 안생김 ### 그리고 저장 폴더 설정은?
                            ### 삭제

                            landmark_dict = dict()
                            for i in ALL:
                                landmark_dict[i] = landmark_list[i]
                            landmark_dict_list.append(landmark_dict)
                            cnt += 1

                        else:
                            os.remove(f)
                            self.del_csv()
                            remove_counts += 1
                        with open(os.path.join(self.base_path, f'json1/rich_{rank}.json'), "w") as json_file:
                            json_file.write(json.dumps(landmark_dict_list))
                            json_file.write('\n')

                    n += 1

                else:
                    os.remove(f)
                    self.del_csv()
                    minus_value += 1
                    remove_counts += 1
            else:
                os.remove(f)
                self.del_csv()
                no_dets += 1
                remove_counts += 1

    def json_to_dict(self, path):
        self.path = path
        with open(path, mode='r') as file:
            dictionary = json.load(file)
        return dictionary

    # rich_list.scv는 어떻게 하지???
    def rich_list(self):
        with open('rich_list.csv', mode='r') as file:
            for line in file:
                field = line.strip().split(',')
                return field

    def dict_to_pkl(self):
        rich_list = self.rich_list()
        for i in range(len(rich_list)):
            df = pd.DataFrame(columns=['right_eye_width', 'right_eye_height', 'right_eye_shape', 'right_eyebrow',
                                       'left_eye_width', 'left_eye_height', 'left_eye_shape', 'left_eyebrow',
                                       'eye_between', 'nose_width', 'nose_height', 'philtrum',
                                       'mouth_width', 'mouth_height', 'mouth_shape', 'jaw_height'])
            for j in glob.glob(os.path.join(self.base_path, f'json1/rich_{101 + i}.json')):
                with open(j, mode='r') as file:
                    lst = json.load(file)
                    for k in range(len(lst)):
                        dc = self.json_to_dict(j)[k]
                        pt2 = FaceDistanceRatio(dc)
                        data = pt2.dict_to_series()
                        df.loc[k] = data
                    print(df)
                    with open(f'../project/dataframe/{101 + i}_{rich_list[i].strip()}.pkl', mode='wb') as pic:
                        pickle.dump(df, pic)

    # 사진이 저장 안된 부자들은 어떻게 할지?????
    def rich_total_dataframe(self):
        df = pd.DataFrame(columns=['right_eye_width', 'right_eye_height', 'right_eye_shape', 'right_eyebrow',
                                   'left_eye_width', 'left_eye_height', 'left_eye_shape', 'left_eyebrow',
                                   'eye_between', 'nose_width', 'nose_height', 'philtrum',
                                   'mouth_width', 'mouth_height', 'mouth_shape', 'jaw_height'])
        rich_list = self.rich_list()

        for i in range(101, 600):
            if os.path.isfile(os.path.join(self.base_path, f'dataframe/{i}_{rich_list[i - 101]}.pkl')) == True:
                with open(os.path.join(self.base_path, f'dataframe/{i}_{rich_list[i - 101]}.pkl'), mode='rb') as f:
                    p = pickle.load(f)
                    df2 = pd.DataFrame(p)
                    sr = df2.mean(axis=0)
                    df.loc[i - 101] = sr
                    df.rename(index={(i - 101): rich_list[i - 101]}, inplace=True)
            else:
                pass
        sr = df.mean(axis=0)
        df.loc['rich_mean'] = sr

        sr_cv = df.std(axis=0) / df.mean(axis=0)
        df.loc['rich_cv'] = sr_cv
        df_type = df.T.sort_values('rich_cv').iloc[1:11].T
        # 0 자수성가형, 1 금수저형, 2 투자의귀재, 3 또라이형, 4 자퇴형
        # 5 결혼형, 6 시인형, 7 UN특사형, 8 정치인형, 9 professor type
        type = [0, 4, 4, 0, 4, 1, 0, 0, 0, 4, 1, 0, 0, 1, 1, 1, 5, 1, 5, 3,
                0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 4, 0, 1, 0, 0, 1, 1, 5, 0, 0,
                1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 5, 1, 0, 1, 1, 0, 1, 0, 0, 8,
                0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 2, 1, 2, 0, 1, 0, 1, 2,
                2, 0, 0, 0, 0, 0, 1, 1, 0, 1, 2, 0, 1, 1, 1, 1, 4, 2, 0, 0,
                0, 0, 2, 0, 5, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 4,
                0, 1, 2, 2, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 1, 1, 4, 1, 0, 0, 4, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 2, 0,
                0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0,
                1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 5, 0, 0, 1, 0, 0,
                1, 0, 1, 0, 4, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                1, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 1, 1, 2, 1, 2, 1, 0, 0,
                1, 1, 2, 0, 0, 1, 0, 1, 0, 2, 2, 4, 0, 5, 0, 0, 0, 0, 0, 0,
                0, 2, 0, 0, 1, 0, 5, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1,
                0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0,
                0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 2, 1, 1, 1, 2, 0, 1, 2, 0, 0,
                0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 2, 1, 0, 7, 1, 1, 0, 1, 1, 1,
                1, 0, 0, 1, 1, 1, 0, 2, 2, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1,
                1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 3, 1, 0, 3, 1, 1,
                1, 0, 1, 1, 0, 0, 0, 3, 0, 0, 1, 1, 1, 1, 6, 0, 0, 1, 1, 0,
                1, 1, 0, 0, 1, 2, 0, 1, 0, 0, 0, 0, 0, 1, 9, 1, 2, 2, 1, 1,
                1, 0, 5, 0, 0, 4, 5, 0, 0, 1, 4, 3, 0, 4, 1, 1, 0, 0, 1, 3,
                0, 0, 2, 2, 2, 0, 1, 1, 0, 4, 0, 1, 1, 1, 1, 1, 0, 1, 1, 3,
                5, 0, 0, 0, 1, 1, 1, 0, 1, 5, 1, 0, 0, 6, 1, 4, 0, 2, 0, 1,
                0, 0, 1, 1, 3, 0, 5, 1, 0, 0, 1, 1, 1, 1, 0, 1, None, None]
        df_type['type'] = type

        return df_type

    def crawling(self):
        """
        사전 준비 작업
        0) 아래 2개의 파일은 미리 다운 받아서 저장 해야함
            01_Subpixel.py
            02_Datagenerator.py
        1) celeba-dataset폴더를 압축 해제 한 후 - 'imageset_test'폴더를 생성한다.
            - 이때 아무 사진이나 1개를 넣어둔다.
        2-1) celeba-dataset폴더 안에 'processed_test'폴더를 생성한다.
        2-2) 2-1에서 생성한 폴더 안에 'x_test', 'y_test'폴더를 각각 생성한다.
        3) celeba-dataset폴더 안에 'list_eval_partition_test.csv'파일을 생성하는데 이때 주의할점
            - 1)에서 넣어둔 사진의 이름을 1번 줄에 적는다(파일의 이름,2) 꼴로
        4) celeba-dataset폴더에 'json1'의 폴더를 생성한다.
        5) celeba-dataset폴더에 'dataframe'폴더를 생성한다.

        구동 방식
        1. keras에 있는 기존 train data를 훈련 시킨다.
        2. 훈련이 완료 되면 url에서 rich_list를 출력하여 list의 형태로 저장한다.
        3. list의 형태로 저장된 rich_list에서 원소를 하나씩 뽑아 구글 이미지 검색을 실시한다.
        4. 이미지 검색후 다운로드한다.
        5. keras로 화질 개선 후 저장한다.
        6. 이때 다운로드 된 사진을 정해진 양식대로 'list_eval_partition_test.csv'에 차례대로 저장한다.
        7. facing_67 함수로 화질 개선이 된 파일을 정리한다.
            - 이미지내의 얼굴을 인식한 뒤 제외 사항(얼굴이 2개 이상인경우, 코의 각도가 일정 각도를 벗어난 경우, 얼굴이 인식되지 않은 경우)인 경우
             'list_eval_partition_test.csv'에서 삭제한다.
            - 삭제 되지 않은 사진은 dlib을 이용하여 68개의 좌표를 dictionary형태의 json으로 저장한다.
        8. 이미지가 제외되지 않으면 이 과정을 rich_list의 원소 하나당 20번 반복한다.
        9. 총 499개의 원소가 모두 반복
        10. json으로 저장된 파일을 pickle형식 DataFrame으로 저장한다.
        11. DataFrame으로 저장된 것을 하나의 DF으로 합치고, type를 추가한 뒤 유의미한 columns만 뽑아서 하나의 DF 생성

        :return: rich_total_dataframe(self)에서 만들어진 하나의 DF
        """
        self.img_resolution_train()
        self.img_resolution_train2()

        url = "https://ceoworld.biz/2020/02/28/rich-list-index-2020/"

        res = requests.get(url)
        html = res.text.strip()
        soup = bs(html, 'html.parser')  # BeautifulSoup -> bs
        rich_link = soup.select('tbody.row-hover td.column-2')

        # 리스트에서 html 코드 제거
        rich_list = [str(rich).replace("</td>", '').replace("<td class=\"column-2\">", '') for rich in rich_link]

        # 리스트에서 띄어쓰기 없애고 대신 url 주소에 띄어쓰기로 먹히는 + 대체
        rich_friends = [rich.replace(' ', '+') for rich in rich_list]

        r10 = rich_friends[:500]  # 상위 10명만

        rank = 101  # 부자 순위대로 나열하기 위한 카운트
        for r in r10:
            url = f'https://www.google.com/search?q={r}&sxsrf=ALeKk01WuCtRoFmDGbZmzgJxG5b6wz8VrQ:1592534710712&' \
                  'source=lnms&tbm=isch&sa=X&ved=2ahUKEwjEtuWN7ozqAhVbIIgKHcJdD9MQ_AUoAXoECBgQAw&biw=1920&bih=1089'
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            html = urlopen(req)
            soup = bs(html, "html.parser")
            img = soup.find_all(class_='t0fcAb')
            n = 1  # 이미지 따로 저장하기 위한 카운트
            for i in img:
                imgUrl = i.attrs['src']
                with urlopen(imgUrl) as f:
                    with open(f"{os.path.join(self.base_path, 'imageset_test/')}" + str(rank) + '_' +
                              r.replace('+', '') + str(n) + '.jpg', mode='wb') as h:  # w - write b - binary
                        img = f.read()
                        h.write(img)
                        with open(os.path.join(self.base_path, 'list_eval_partition_test.csv'),
                                  newline='', mode='a') as file:
                            wr = csv.writer(file)
                            files = sorted(glob.glob(os.path.join(self.base_path, 'imageset_test', '*.jpg')),
                                           key=os.path.getctime,
                                           reverse=True)
                            len_path = len(glob.glob(os.path.join(self.base_path, 'imageset_test'))[0]) + 1
                            wr.writerow([files[0][len_path:], 2])

                    src = cv2.imread(
                        str(f"{os.path.join(self.base_path, 'imageset_test')}/{str(rank)}_{r.replace('+', '')}{str(n)}.jpg"),
                        cv2.IMREAD_COLOR)
                    print(f"{os.path.join(self.base_path, 'imageset_test')}/{str(rank)}_{r.replace('+', '')}{str(n)}.jpg")
                    resizing = cv2.resize(src, dsize=(178, 218), interpolation=cv2.INTER_CUBIC)
                    gtorgb = cv2.cvtColor(resizing, cv2.COLOR_BGR2RGB)
                    imageio.imwrite(
                        f"{os.path.join(self.base_path, 'imageset_test')}/{str(rank)}_{r.replace('+', '')}{str(n)}.jpg",
                        im=gtorgb, pilmode='CMYK', as_gray=False)

                    self.img_resolution_test()
                    self.img_resolution_test2()
                    self.facing_67(face_folder_path=os.path.join(self.base_path, 'imageset_test'), rank=rank)

                n += 1
                if n > 30:  # 10장만 출력하기
                    break
            rank += 1
        self.dict_to_pkl()
        return self.rich_total_dataframe()


if __name__ == '__main__':
    cr = CrawlingRich(r"C:\labs\\project\\project\\celeba-dataset")
    # glob를 붙이면 list형식으로 주소 출력
    # glob를 안붙이면 문자열로 출력
    print(glob.glob(os.path.join(r"C:\labs\\project\\project\\celeba-dataset", 'list_eval_partition_test.csv'))[0])
    print(glob.glob(os.path.join(r"C:\labs\\project\\project\\celeba-dataset", 'list_eval_partition_test.csv')))
    print(glob.glob(os.path.join("C:\labs\\project\\project\\celeba-dataset", 'list_eval_partition_test.csv')))
    print(os.path.join("C:\labs\\project\\project\\celeba-dataset", 'list_eval_partition_test.csv'))

    path = r"C:\labs\\project\\project\\celeba-dataset"
    files = glob.glob(os.path.join(path, "imageset_test", "101_JeffBezos1.jpg"))
    len_path = len(glob.glob(os.path.join(path, "imageset_test"))[0]) + 1
    print(len_path)
    print(files[0][len_path:])

    # print(cr.crawling())