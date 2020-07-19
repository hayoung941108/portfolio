import math
import pandas as pd
import csv
import json
import requests
from bs4 import BeautifulSoup as bs


def rich_index(url="https://ceoworld.biz/2020/02/28/rich-list-index-2020/"):
    res = requests.get(url)
    html = res.text.strip()
    soup = bs(html, 'html.parser')  # BeautifulSoup -> bs
    rich_link = soup.select('tbody.row-hover td.column-2')
    rich_list = [str(rich).replace("</td>", '').replace("<td class=\"column-2\">", '') for rich in rich_link]

    return rich_list

def rich_list():
    with open('rich_list.csv', mode='r') as file:
        for line in file:
            field = line.strip().split(',')
            return field


class FaceDistanceRatio:
    """
    얼굴에서 구할 수 있는 길이
    1) 눈의 가로/세로 길이
        오른쪽 눈 가로 좌표 : 36, 39
        오른쪽 눈 세로 좌표 : 37, 41
        왼 쪽 눈 가로 좌표 : 42, 45
        왼 쪽 눈 세로 좌표 : 43, 47
    2) 미간의 길이 : 39, 42
    3) 코의 가로/세로 길이
        코의 가로 좌표 : 31, 35
        코의 세로 좌표 : 27, 30
    4) 인중의 길이 : 33, 51
    5) 입의 가로 세로 길이
        입의 가로 좌표 : 48, 54
        입의 세로 좌표 : 51, 57
    6) 하관의 길이 : 57, 8
    7) 눈썹의 길이
        오른쪽 눈썹 좌표 : 17, 18, 19 20
        왼 쪽 눈썹 좌표 : 22, 23, 24, 25
    8) 눈꼬리의 높낮이
        오른쪽 : 39(눈안쪽) 대비 36
        왼 쪽 : 42(눈안쪽) 대비 45
    9) 입꼬리의 높낮이 : (62, 66)의 y좌표 값의 중앙점 기준 - 아래, 중앙, 위

    기준으로 생각할 수 있는 길이
    1) 광대에서 광대 까지의 거리 : 2, 15
    2) 얼굴의 세로 길이 : (21, 22)의 중앙점, 8

    """

    def __init__(self, dictionary):
        self.dictionary = dictionary

    def distance(self, a, b):
        self.a = self.dictionary[a]
        self.b = self.dictionary[b]
        x1, y1, x2, y2 = self.a[0], self.a[1], self.b[0], self.b[1]
        distance = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        return distance

    # 얼굴의 세로 길이
    def face_height(self, a='21', b='22', c='8'):
        self.a = self.dictionary[a]
        self.b = self.dictionary[b]
        self.c = self.dictionary[c]
        x1, y1, x2, y2 = self.a[0], self.a[1], self.b[0], self.b[1]
        self.dictionary['ab'] = [(x1 + x2) / 2, (y1 + y2) / 2]
        return self.distance('ab', c)

    # 얼굴의 가로 길이(광대에서 광대까지의 거리)
    def face_width(self, a='2', b='15'):
        return self.distance(a, b)

    # 얼굴의 가로 길이 대비 오른쪽 눈의 가로 길이 비
    def right_eye_width(self, a='36', b='39'):
        return self.distance(a, b) / self.face_width()

    # 얼굴의 세로 길이 대비 오른쪽 눈의 세로 비
    def right_eye_height(self, a='37', b='41'):
        return self.distance(a, b) / self.face_height()

    # 얼굴의 가로 길이 대비 오른쪽 눈의 가로 길이 비
    def left_eye_width(self, a='42', b='45'):
        return self.distance(a, b) / self.face_width()

    # 얼굴의 세로 길이 대비 오른쪽 눈의 세로 비
    def left_eye_height(self, a='43', b='47'):
        return self.distance(a, b) / self.face_height()

    # 얼굴 가로 길이 대비 미간 길이의 비
    def eye_between(self, a='39', b='42'):
        return self.distance(a, b) / self.face_width()

    # 코의 가로 길이 비
    def nose_width(self, a='31', b='35'):
        return self.distance(a, b) / self.face_width()

    # 코의 세로 길이 비
    def nose_height(self, a='27', b='30'):
        return self.distance(a, b) / self.face_height()

    # 인중의 길이 비
    def philtrum(self, a='33', b='51'):
        return self.distance(a, b) / self.face_height()

    # 입의 가로 길이 비
    def mouth_width(self, a='48', b='54'):
        return self.distance(a, b) / self.face_width()

    # 입의 세로 길이 비
    def mouth_height(self, a='51', b='57'):
        return self.distance(a, b) / self.face_height()

    # 하관의 길이 비
    def jaw_height(self, a='57', b='8'):
        return self.distance(a, b) / self.face_height()

    # 오른쪽 눈썹의 길이 비
    def right_eyebrow(self, a='17', b='18', c='19', d='20'):
        tot_distanse = self.distance(a, b) + self.distance(b, c) + self.distance(c, d)
        return tot_distanse / self.face_width()

    # 왼쪽 눈썹의 길이 비
    def left_eyebrow(self, a='22', b='23', c='24', d='25'):
        tot_distanse = self.distance(a, b) + self.distance(b, c) + self.distance(c, d)
        return tot_distanse / self.face_width()

    # 오른쪽 눈꼬리의 높낮이
    # 언제가 높은지 낮은지 정의 해주어야함(논의 필요)
    def right_eye_shape(self, a='39', b='36'):
        self.a = self.dictionary[a]
        self.b = self.dictionary[b]
        # if abs(self.a[1] - self.b[1]) <= 3.5:
        #     return 0
        # else:
        #     if self.a[1] - self.b[1] < -3.5:
        #         return -1
        #     else:
        #         return 1
        return (self.a[1] - self.b[1]) / self.face_height()

    # 왼쪽 눈꼬리의 높낮이
    def left_eye_shape(self, a='42', b='45'):
        self.a = self.dictionary[a]
        self.b = self.dictionary[b]
        # if abs(self.a[1] - self.b[1]) <= 3.5:
        #     return 0
        # else:
        #     if self.a[1] - self.b[1] < -3.5:
        #         return -1
        #     else:
        #         return 1
        return (self.a[1] - self.b[1]) / self.face_height()

    # 입꼬리 높낮이
    def mouth_shape(self, a='48', b='54', c='62', d='66'):
        # 1) 입의 양쪽 끝부분(48, 54) 중점의 y좌표를 찾는다.
        # 2) 입술의 가장 윗부분 62의 y좌표, 입술의 가장 아랫부분 66의 y좌표
        #    중앙점을 찾는다.
        # 3) 1, 2에서 구한 y좌표의 차를 계산한다.
        #       값이 기준값사이에 있다면 입꼬리가 수평에 가까운 것
        #       값이 기준값보다 작으면 입꼬리가 내려간 것
        #       값이 기준값보가 크면 입꼬리가 올라간 것
        self.a = self.dictionary[a]
        self.b = self.dictionary[b]
        self.c = self.dictionary[c]
        self.d = self.dictionary[d]
        mid_ab_y = (self.a[1] + self.b[1]) / 2  # 1) 구한 값
        mid_cd_y = (self.c[1] + self.d[1]) / 2  # 2) 구한 값
        # if abs(mid_ab_y - mid_cd_y) <= 3.5:
        #     return 0
        # else:
        #     if mid_ab_y - mid_cd_y < -3.5:
        #         return -1
        #     else:
        #         return 1
        return (mid_ab_y - mid_cd_y) / self.face_height()

    # 사진에서 구할 수 있는 모든 값들의 정보
    def info(self):
        print('오른쪽 눈의 가로 비:', self.right_eye_width())
        print('오른쪽 눈의 세로 비:', self.right_eye_height())
        print('오른쪽 눈꼬리의 모양:', self.right_eye_shape())
        print('오른쪽 눈썹의 비', self.right_eyebrow())
        print('왼쪽 눈의 가로 비:', self.left_eye_width())
        print('왼쪽 눈의 세로 비:', self.left_eye_height())
        print('왼쪽 눈꼬리의 모양:', self.left_eye_shape())
        print('왼쪽 눈썹의 비', self.left_eyebrow())
        print('미간의 길이 비:', self.eye_between())
        print('코의 가로 비:', self.nose_width())
        print('코의 세로 비:', self.nose_height())
        print('인중의 길이 비:', self.philtrum())
        print('입의 가로 비:', self.mouth_width())
        print('입의 세로 비:', self.mouth_height())
        print('입꼬리의 모양:', self.mouth_shape())
        print('하관 비:', self.jaw_height())
        print()

    def dict_to_df(self, row_names):
        df = pd.DataFrame(
            data=[[self.right_eye_width(), self.right_eye_height(), self.right_eye_shape(), self.right_eyebrow(),
                  self.left_eye_width(), self.left_eye_height(), self.left_eye_shape(), self.left_eyebrow(),
                  self.eye_between(), self.nose_width(), self.nose_height(), self.philtrum(),
                  self.mouth_width(), self.mouth_height(), self.mouth_shape(), self.jaw_height()]],
            index=[row_names],
            columns=['right_eye_width', 'right_eye_height', 'right_eye_shape', 'right_eyebrow',
                     'left_eye_width', 'left_eye_height', 'left_eye_shape', 'left_eyebrow',
                     'eye_between', 'nose_width', 'nose_height', 'philtrum',
                     'mouth_width', 'mouth_height', 'mouth_shape', 'jaw_height'])
        return df

    def dict_to_series(self):
        series = [self.right_eye_width(), self.right_eye_height(), self.right_eye_shape(), self.right_eyebrow(),
                  self.left_eye_width(), self.left_eye_height(), self.left_eye_shape(), self.left_eyebrow(),
                  self.eye_between(), self.nose_width(), self.nose_height(), self.philtrum(),
                  self.mouth_width(), self.mouth_height(), self.mouth_shape(), self.jaw_height()]

        return series


if __name__ == '__main__':
    diction = {"0": [88, 225], "1": [90, 257], "2": [93, 289], "3": [98, 321], "4": [106, 351], "5": [123, 378],
               "6": [144, 402], "7": [170, 420], "8": [199, 425], "9": [227, 420], "10": [252, 402], "11": [274, 379],
               "12": [290, 352], "13": [299, 323], "14": [306, 292], "15": [311, 260], "16": [312, 228],
               "17": [105, 212], "18": [118, 197], "19": [139, 195], "20": [159, 199], "21": [178, 208],
               "22": [215, 207], "23": [234, 199], "24": [254, 195], "25": [275, 198], "26": [290, 213],
               "27": [196, 232], "28": [196, 255], "29": [196, 277], "30": [195, 301], "31": [173, 311],
               "32": [183, 314], "33": [195, 318], "34": [207, 314], "35": [219, 310], "36": [129, 233],
               "37": [140, 227], "38": [154, 228], "39": [165, 237], "40": [152, 238], "41": [139, 237],
               "42": [228, 237], "43": [239, 228], "44": [253, 227], "45": [265, 234], "46": [254, 238],
               "47": [240, 238], "48": [153, 342], "49": [169, 337], "50": [184, 333], "51": [196, 336],
               "52": [208, 333], "53": [223, 336], "54": [242, 341], "55": [224, 357], "56": [209, 365],
               "57": [196, 367], "58": [183, 365], "59": [168, 357], "60": [160, 344], "61": [184, 345],
               "62": [196, 346], "63": [208, 345], "64": [234, 343], "65": [208, 346], "66": [196, 348],
               "67": [184, 346]}
    pt1 = FaceDistanceRatio(diction)
    pt1.info()
    print(pt1.dict_to_df(['dong']))

    with open('rich_list.csv', mode='w') as f:
        wr = csv.writer(f)
        files = rich_index()
        wr.writerow(files)
    print(rich_list())

    print(pt1.dict_to_series())

