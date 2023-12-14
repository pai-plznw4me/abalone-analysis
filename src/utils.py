import re
import os
from PIL import Image
import shutil
import random
from xml.etree.ElementTree import parse
import os.path
import numpy as np
import cv2
import tifffile as tifi
import math
import matplotlib.pyplot as plt
import json
import pandas as pd
from glob import glob
from tqdm import tqdm
from pyproj import Transformer
import rasterio
from shapely.geometry import Polygon


def paths2imgs(paths, resize=None, error=None):
    """
    Description:

    :param paths: list, [str, str, ... str], 이미지 저장 경로
    :param resize: tuple, (h, w)
    :param error: 잘못된 경로를 반환합니다.
    :return:
    """
    imgs = []
    error_paths = []
    for path in paths:
        try:
            img = path2img(path, resize)
            imgs.append(img)
        except:
            print(os.path.exists(path))
            print("{} 해당 경로에 파일이 존재하지 않습니다.".format(path))
            error_paths.append(path)

    if error == 'error_return':
        return imgs, error_paths

    return imgs


def path2img(path, resize=None):
    """
    Description:
    경로에 있는 이미지를 RGB 컬러 형식으로 불러옵니다
    resize 에 값을 주면 해당 크기로 이미지를 불러옵니다.
    :param path: str
    :param resize: tuple or list , (W, H)
    return img, ndarray, shape=(h, w, 3)
    """

    # 경로 중 한글명이 있어 cv2 만으로는 읽을 수 없었기 때문에, numpy 로 파일을 읽은 후 이를 cv2.imdecode 를 통해 이미지로 변환합니다.
    # HEIC 경로인 경우 아래와 경로로 작업 합니다.
    if path.endswith('heic') or path.endswith('HEIC'):
        img = np.array(Image.open(path))
    else:
        img = cv2.imread(path)

        # BGR 을 RGB 로 변환합니다.
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if resize:
        h, w = resize
        img = cv2.resize(img, (w, h))

    return img


def tifs2exts(src_dir, dst_dir, ext):
    """
    Description:
        여러장의 tif 파일을 지정된 확장자로 변환후 저장합니다.(저장만 지원합니다)

    :param str src_dir:
    :param str dst_dir:
    :param str ext:
    """

    src_paths = all_image_paths(src_dir)
    assert len(src_paths) >= 0, '개수가 0보다 커야 합니다.'
    os.makedirs(dst_dir, exist_ok=True)
    for src_path in src_paths:
        try:
            _ = tif2ext(src_path, dst_dir, ext)
        except:
            print('Error image: {}'.format(src_path))


def tif2ext(src_path, dst_dir, ext):
    """
    tif 파일을 지정된 확장자로 변환해 저장합니다.

    :param str src_path:
    :param str dst_dir:
    :param str ext: jpg or .jpg
    """

    # dst path 을 생성합니다.
    ext = ext.replace('.', '').lower()
    name = get_name(src_path, ext=False)
    os.makedirs(dst_dir, exist_ok=True)
    dst_path = os.path.join(dst_dir, name + '.' + ext)

    # TIF 이미지 열기
    with Image.open(src_path) as img:
        # 변환된 이미지 저장
        # cv2.imwrite(dst_path, np.array(img))
        if ext == 'png':
            img.save(dst_path, format='PNG')
        elif ext == 'jpg' or ext == 'jpeg':
            img.save(dst_path, format='JPEG')
        else:
            print('지원하지 않는 확장자 입니다.')


def subdivide(src_dir, dst_dir, n_files, copy, shuffle=True):
    """
    Description
        폴더 안에 들어 있는 데이터를 소분합니다.

    Args:
        :param str src_dir: 폴더 별 파일 개 수
        :param str dst_dir: 저장 할 폴더
        :param int n_files: 서브 폴더 별 파일 개 수
         :param bool copy: 복사 여부
            True 시 파일 복사
            False 시 파일 이동(move)
        :param bool shuffle: 경로를 무작위로 섞음

    Usage:
        import os
        import shutil
        from utility import subdivide

        # 데이터를 코드별로 분류 하기
        root_src_dir = '../../datasets/unify_imgs_256'
        root_dst_dir = '../../datasets/codes/unify_imgs_256_subdivide'

        os.makedirs(root_dst_dir, exist_ok=True)
        codes = os.listdir(root_src_dir)

        for code in codes[:]:
            src_dir = os.path.join(root_src_dir, code)
            dst_dir = os.path.join(root_dst_dir, code)
            subdivide(src_dir, dst_dir, 1000, False, True)
    """

    file_paths = all_file_paths(src_dir)
    n_paths = len(file_paths)
    if shuffle:
        random.shuffle(file_paths)

    n_subfolders = int(np.ceil(n_paths / n_files))

    for ind in range(n_subfolders):

        # 서브 폴더를 생성합니다. 서브 폴더 이름은 인덱스로 합니다.
        subfolder_name = str(ind)
        subdir = os.path.join(dst_dir, subfolder_name)
        os.makedirs(os.path.join(dst_dir, str(subfolder_name)), exist_ok=True)

        # 전체 경로를 소분합니다.
        slc = slice(n_files * ind, n_files * (ind + 1))
        sliced_file_paths = file_paths[slc]

        # 파일을 서브폴더로 이동합니다.
        sliced_names = get_names(sliced_file_paths, ext=True)
        for src_path, name in zip(sliced_file_paths, sliced_names):
            dst_path = os.path.join(subdir, name)
            if copy:
                shutil.copy(src_path, dst_path)
            else:
                shutil.move(src_path, dst_path)


def get_names(paths, ext=True):
    names = [get_name(path, ext) for path in paths]
    return names


def get_name(path, ext=True):
    if ext:
        return os.path.split(path)[-1]
    else:
        return os.path.splitext(os.path.split(path)[-1])[0]


def get_ext(path):
    return os.path.splitext(path)[-1]


def get_exts(paths):
    return [get_ext(path) for path in paths]


def all_image_paths(image_dir):
    """
    Description:
        입력 된 폴더 하위 경로의 모든 이미지 파일 경로를 찾아내 반환합니다.
    Args:
        :param str image_dir: 이미지 폴더

    Returns: list, [str, str ... str]
    """
    file_paths = all_file_paths(image_dir)
    img_paths = filter_img_paths(file_paths)
    return img_paths


def all_file_paths(image_dir):
    """
    Description:
        입력 된 폴더 하위 경로의 모든 파일 경로를 찾아내 반환합니다.
    Args:
        :param str image_dir: 파일 폴더

    Returns: list, [str, str ... str]
    """
    paths = []
    for folder, subfolders, files in os.walk(image_dir):
        for file in files:
            path = os.path.join(folder, file)
            paths.append(path)
    return paths


def filter_img_paths(paths):
    """
    Description:
        입력 경로중 '.gif|.jpg|.jpeg|.tiff|.png' 해당 확장자를 가진 파일만 반환합니다.
    Args:
    :param: paths, [str, str, str ... str]
    :return: list, [str, str, str ... str]
    """
    regex = re.compile("(.*)(\w+)(.gif|.jpg|.jpeg|.tiff|.png|.bmp|.JPG|.HEIC|.tif)")
    img_paths = []
    for path in paths:
        if regex.search(path):
            img_paths.append(path)
    return img_paths


def get_pascal_bbox(filepath):
    """
    Description:
        pascal voc format 에서 bounding box 정보를 추출합니다.
    Args:
    :list return:
        [[x1, y1, x2, y2], [x1, y1, x2, y2] ... [x1, y1, x2, y2]]
    """
    tree = parse(filepath)
    root = tree.getroot()
    coords = []
    for obj in root.iter('object'):
        xmin = int(obj.find('bndbox').findtext('xmin'))
        xmax = int(obj.find('bndbox').findtext('xmax'))
        ymin = int(obj.find('bndbox').findtext('ymin'))
        ymax = int(obj.find('bndbox').findtext('ymax'))
        print(xmin, xmax, ymin, ymax)
        coords.append([xmin, ymin, xmax, ymax])
    return coords


def read_tiff_with_resize(filepath, resize_ratio=1.0):
    """
    Description:
        tiff file 을 읽어 반환합니다. 지정된 ratio 형태로 이미지를 resize 하여 반환합니다.
    Args:
        :param filepath:
        :param resize_ratio:
    :return numpy.ndarray img_resized: ndarray 형태로 반환합니다.
    """
    # tifffile open
    img = tifi.imread(filepath)
    dsize = (np.array(img.shape[:2]) * resize_ratio).astype(int)
    if not resize_ratio == 1.0:
        img_resized = cv2.resize(img, dsize=dsize[::-1])
    else:
        img_resized = img
    return img_resized


def save_image(obj, savepath):
    """
    Description:
    numpy 이미지를 저장합니다.

    Args:
        :param str savepath: 저장 경로

    """
    # save resized image
    cv2.imwrite(filename=savepath, img=obj)

    if not os.path.exists(savepath):
        print('파일이 저장되지 않았습니다. 지정된 경로 : {}'.format(savepath))


def save_imgs(dst_paths, src_imgs):
    """
    :param dst_paths: list = [str, str, ... str]
    :param src_imgs: ndarray
    :return:
    """
    for path, img in tqdm(zip(dst_paths, src_imgs)):
        Image.fromarray(img).save(path)


def plot_images(imgs, names=None, random_order=False, savepath=None, show=True, figsize=None, fontsize=10):
    h = math.ceil(math.sqrt(len(imgs)))
    fig = plt.figure(figsize=figsize)
    plt.gcf().set_size_inches((20, 20))
    for i in range(len(imgs)):
        ax = fig.add_subplot(h, h, i + 1)
        if random_order:
            ind = random.randint(0, len(imgs) - 1)
        else:
            ind = i
        try:
            img = imgs[ind]
            plt.axis('off')
            if len(img.shape) == 2:
                plt.imshow(img, 'gray')
            else:
                plt.imshow(img)
            # save image
        except:
            img = np.zeros(shape=(50, 50, 3))
            plt.axis('off')
            plt.imshow(img)
            if len(img.shape) == 2:
                plt.imshow(img, 'gray')
            else:
                plt.imshow(img)
            # save image
        if not names is None:
            ax.set_title(str(names[ind]), c='red', size=fontsize)
    if not savepath is None:
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        plt.savefig(savepath)
    if show:
        plt.tight_layout()
        plt.show()


def paths2imgs(paths, resize=None, error=None):
    """
    Description:
        경로을 numpy 로 변환해 반환합니다.
    Args:
        :param paths: list, [str, str, ... str], 이미지 저장 경로
        :param resize: tuple, (h, w)
        :param error: 잘못된 경로를 반환합니다.
    :list return:
    """
    imgs = []
    error_paths = []
    for path in paths:
        try:
            img = path2img(path, resize)
            imgs.append(img)
        except:
            print(os.path.exists(path))
            print("{} 해당 경로에 파일이 존재하지 않습니다.".format(path))
            error_paths.append(path)

    if error == 'error_return':
        return imgs, error_paths

    return imgs


def path2img(path, resize=None):
    """
    Description:
        경로에 있는 이미지를 RGB 컬러 형식으로 불러옵니다
        resize 에 값을 주면 해당 크기로 이미지를 불러옵니다.
    Args:
        :param path: str
        :param resize: tuple or list , (W, H)
    :return np.ndarray img: shape=(h, w, 3)
    """

    # 경로 중 한글명이 있어 cv2 만으로는 읽을 수 없었기 때문에, numpy 로 파일을 읽은 후 이를 cv2.imdecode 를 통해 이미지로 변환합니다.
    # HEIC 경로인 경우 아래와 경로로 작업 합니다.
    if path.endswith('heic') or path.endswith('HEIC'):
        img = np.array(Image.open(path))

    else:
        img = cv2.imread(path)

        # BGR 을 RGB 로 변환합니다.
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if resize:
        h, w = resize
        img = cv2.resize(img, (w, h))

    return img


def polygon_center_xy(*polygon_points):
    """
    Description:
        polygon 내 center 좌표  값을 찾아 반환합니다.
    Args:
        :param polygon_points:
    :return:
    """
    x_coordinates, y_coordinates = zip(*polygon_points)
    center_x = sum(x_coordinates) / len(polygon_points)
    center_y = sum(y_coordinates) / len(polygon_points)
    return center_x, center_y


def write_msg_in_polygon_center(img, text, *polygon_points, **kwargs):
    """
    Description:
        폴리곤 중앙에 지정된 메시지를 그립니다.

    Args:
        :param str img: 이미지에 입력할 텍스트
        :param str text: 이미지에 입력할 텍스트
        :param list polygon_points: [(x1, y1),(x2, y2) ... (x3, y3)]

    :return:
    """

    cx, cy = polygon_center_xy(*polygon_points)
    position = int(cx), int(cy)

    # 텍스트를 쓸 위치와 내용을 정의합니다.
    if not 'font' in kwargs.keys():
        font = cv2.FONT_HERSHEY_SIMPLEX
    else:
        font = kwargs['font']  # 글꼴 크기

    if not 'font_scale' in kwargs.keys():
        font_scale = 3  # 글꼴 크기
    else:
        font_scale = kwargs['font_scale']

    if not 'font_color' in kwargs.keys():
        font_color = (0, 0, 255)  # 글꼴 색상 (BGR 형식)
    else:
        font_color = kwargs['font_color']

    if not 'font_thickness' in kwargs.keys():
        font_thickness = 10  # 글꼴 두께
    else:
        font_thickness = kwargs['font_thickness']

    img = cv2.putText(img, text, position, font, font_scale, font_color, font_thickness)
    return img


def extract_pixel_in_polygon(image, coords):
    """
    Description:
        polygon 내 존재하는 모든 픽셀 정보를 가져와 반환합니다.

    Args:
        :param np.ndarray image:
        :param list coords: [(x1, y1),(x2, y2) ... (x3, y3)] or [(x1, y1 ,x2, y2, ... ,x_n, y_n]]
    :return:
    """
    canvas = np.zeros(shape=image.shape[:2], dtype=np.uint8)
    resized_coords = np.array(coords)
    resized_coords = resized_coords.reshape((-1, 1, 2)).astype(int)
    mask = cv2.fillPoly(canvas, [resized_coords], color=1)

    # polyline 을 포함하는 가장 작은 직사각형의 좌표를 추출합니다.
    """
    np.argwhere(mask) =  [[ 752 2298]
                          [ 752 2299]
                             ...
                          [ 752 2300]]
    """
    l_y = np.argwhere(mask)[:, 0].min()
    r_y = np.argwhere(mask)[:, 0].max()
    l_x = np.argwhere(mask)[:, 1].min()
    r_x = np.argwhere(mask)[:, 1].max()

    # 원본 이미지에서 직사각형 부분만을 가져옵니다.
    cropped_polylined_img = image[l_y: r_y, l_x: r_x]

    # 마스크에서 직사각형 부분만을 가져옵니다.
    cropped_mask = mask[l_y: r_y, l_x: r_x]

    # 원본 이미지에서 마스크 영역만 추출합니다.
    masked_cropped_polylined_img = cropped_polylined_img * np.expand_dims(cropped_mask, -1)
    return masked_cropped_polylined_img


def crop(image, bbox):
    """

    Parameters
    ----------
    image: ndarray
        numpy 이미지
    bboxes: ndarray, shape
        (x1, y1, x2, y2 )형태 numpy 배열

    Returns
    -------

    """
    image = np.array(image)
    x1, y1, x2, y2 = np.array(bbox).astype(int)
    return image[y1: y2, x1:x2]


def visualize_bboxes(image, bboxes, show=True, show_count=False, color=(0, 255, 0)):
    """

    Parameters
    ----------
    image: ndarray
        numpy 이미지
    bboxes: ndarray, shape
        (x1, y1, x2, y2 )형태 numpy 배열
    Returns
    -------

    """
    # 데이터 시각화
    bboxes = np.array(bboxes).astype(int)
    if show_count:
        bboxes = tqdm(bboxes)
    for sample_coord in bboxes:
        sample_coord.astype('int')
        color = color  # 초록색 (BGR 색상 코드)
        thickness = 1  # 선 두께
        image = cv2.rectangle(image.astype(np.uint8), (sample_coord[0], sample_coord[1]),
                              (sample_coord[2], sample_coord[3]), color,
                              thickness)
    if show:
        plt.imshow(image)
        plt.show()
    return image


def cxcywh_to_xyxy(coord):
    """
    Convert cx, cy, w, h to xyxy format.

    Parameters:
    ----------
    cx : float
        X-coordinate of the center.
    cy : float
        Y-coordinate of the center.
    w : float
        Width of the rectangle.
    h : float
        Height of the rectangle.

    Returns:
    -------
    tuple
        A tuple (x1, y1, x2, y2) representing the top-left and bottom-right
        coordinates of the rectangle in xyxy format.
    """
    cx, cy, w, h = coord
    x1 = cx - (w / 2)
    y1 = cy - (h / 2)
    x2 = cx + (w / 2)
    y2 = cy + (h / 2)

    return x1, y1, x2, y2


def x1y1wh_to_xyxy(coord):
    """
    Convert x1, y1, w, h to xyxy format.

    Parameters:
    ----------
    x : float
        X-coordinate of the left top.
    y : float
        Y-coordinate of the left top.
    w : float
        Width of the rectangle.
    h : float
        Height of the rectangle.

    Returns:
    -------
    tuple
        A tuple (x1, y1, x2, y2) representing the top-left and bottom-right
        coordinates of the rectangle in xyxy format.
    """
    x1, y1, w, h = coord
    x2 = x1 + w
    y2 = y1 + h
    return x1, y1, x2, y2


import numpy as np
from itertools import zip_longest
from numpy.lib.stride_tricks import as_strided


def sliding_window_lastaxis(a, ws, ss):
    """Create a sliding window along the last axis of the array.
    Parameters
    ----------
    a: numpy.ndarray
        The array from which to extract the window
    ws: int
        The size of the window along the dimension of the last axis
    ss: int
        The stride size; how much to move in between windows
        For examples, if `ss = 1`, then the windows are offset by a single
        pixel, whereas if `ss = ws`, the windows are non-overlapping, with a
        new window "starting" immediately after the previous one.
    """
    if ws < 1:
        raise ValueError("`ws` must be at least 1.")
    if ws > a.shape[-1]:
        raise ValueError("`ws` must be shorter than the last axis.")

    ns = 1 + (a.shape[-1] - ws) // ss
    shape = a.shape[:-1] + (ns, ws)
    strides = a.strides[:-1] + (a.strides[-1] * ss, a.strides[-1])
    return as_strided(a, shape=shape, strides=strides)


def sliding_window(a, window, offset, dense=False):
    """
    Create windows/patches into an array.
    Parameters
    ----------
    a : numpy.ndarray
        The array from which to extract windows.
    window: int or iterable
        The window size.
    offset: int or iterable
        The distance to move between windows.
    dense: bool
        If `offset` has fewer elements than `window` and `dense` is set to True,
        the patches returned will be offset by one in between each other when
        `offset` is not specified.
        Otherwise, if `dense` is False (the default), they will be offset by the
        size of the window in that dimension (non-overlapping).
    """
    if not hasattr(window, '__iter__'):
        window = [window]
    if not hasattr(offset, '__iter__'):
        offset = [offset]

    for i, (win, off) in enumerate(zip_longest(window, offset)):
        if off is None:
            if dense:
                off = 1
            else:
                off = win
        a = a.swapaxes(i, -1)
        a = sliding_window_lastaxis(a, win, off)
        a = a.swapaxes(-2, i)
    return a


def ccwh2xyxy(ccwh_boxes):
    cxs = ccwh_boxes[:, 0]
    cys = ccwh_boxes[:, 1]
    ws = ccwh_boxes[:, 2]
    hs = ccwh_boxes[:, 3]

    x1s = cxs - ws // 2
    x2s = cxs + ws // 2
    y1s = cys - hs // 2
    y2s = cys + hs // 2

    return np.stack([x1s, y1s, x2s, y2s], axis=-1)


def get_ious(pred_coords, true_coord):
    """
    Descriptions:
    복수개의 좌표들에 예측 좌표에 대한 하나의 true coord와 iou 을 계산합니다.
    :param pred_coords: ndarray, shape (N, 4)
    :param true_coords: ndarray, shape (1, 4)
    :return: ious: ndarray, shape (N, 4)
    """
    ious = []
    pred_coords = pred_coords.reshape(pred_coords.shape[1] * pred_coords.shape[2], 4)
    for coord in pred_coords:
        # convert (cx, cy, h, w) -> (x1, y1, x2, y2)
        pred_cx, pred_cy, pred_h, pred_w = coord
        pred_x1, pred_y1, pred_x2, pred_y2 = pred_cx - pred_w / 2, pred_cy - pred_h / 2, pred_cx + pred_w / 2, pred_cy + pred_h / 2

        true_cx, true_cy, true_h, true_w = true_coord
        true_x1, true_y1, true_x2, true_y2 = true_cx - true_w / 2, true_cy - true_h / 2, true_cx + true_w / 2, true_cy + true_h / 2

        # calculate each box area
        pred_box_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
        true_box_area = (true_x2 - true_x1) * (true_y2 - true_y1)

        # get coord of inter area
        inter_x1 = max(pred_x1, true_x1)
        inter_y1 = max(pred_y1, true_y1)
        inter_x2 = min(pred_x2, true_x2)
        inter_y2 = min(pred_y2, true_y2)

        # calculate inter box w, h
        inter_w = inter_x2 - inter_x1
        inter_h = inter_y2 - inter_y1

        # calculate inter box area
        inter_area = inter_w * inter_h
        iou = inter_area / (pred_box_area + true_box_area - inter_area)

        ious.append([iou])
    return ious


def xyxy2xywh(xyxy):
    """
    Description:
    x1 y1 x2 y2 좌표를 cx cy w h 좌표로 변환합니다.

    :param xyxy: shape (..., 4), 2차원 이상의 array 가 들어와야 함
    :return: xywh shape(... , 4), 2차원 이상의 array 가 들어와야 함
    """
    w = xyxy[..., 2:3] - xyxy[..., 0:1]
    h = xyxy[..., 3:4] - xyxy[..., 1:2]
    cx = xyxy[..., 0:1] + w * 0.5
    cy = xyxy[..., 1:2] + h * 0.5
    xywh = np.concatenate([cx, cy, w, h], axis=1)
    return xywh


def xywh2xyxy(xywh):
    """
    Description:
    cx cy w h 좌표를 x1 y1 x2 y2 좌표로 변환합니다.

    center x, center y, w, h 좌표계를 가진 ndarray 을 x1, y1, x2, y2 좌표계로 변경
    xywh : ndarary, shape, (..., 4), 마지막 차원만 4 이면 작동.
    """
    cx = xywh[..., 0]
    cy = xywh[..., 1]
    w = xywh[..., 2]
    h = xywh[..., 3]

    x1 = cx - (w * 0.5)
    x2 = cx + (w * 0.5)
    y1 = cy - (h * 0.5)
    y2 = cy + (h * 0.5)

    return np.stack([x1, y1, x2, y2], axis=-1)


def draw_rectangle(img, coordinate, color=(255, 0, 0)):
    """
    Description:
    img 에 하나의 bounding box 을 그리는 함수.

    :param img: ndarray 2d (gray img)or 3d array(color img),
    :param coordinate: tuple or list(iterable), shape=(4,) x1 ,y1, x2, y2
    :param color: tuple(iterable), shape = (3,)
    :return:
    """

    # opencv 에 입력값으로 넣기 위해 반드시 정수형으로 변경해야함
    coordinate = coordinate.astype('int')
    x_min = coordinate[0]
    x_max = coordinate[2]
    y_min = coordinate[1]
    y_max = coordinate[3]

    img = img.astype('uint8')

    return cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color)


def draw_rectangles(img, coordinates, color=(255, 0, 0)):
    """
    Description:
    하나의 img 에 복수개의  bounding box 을 그리는 함수.

    :param img: ndarray 2d (gray img)or 3d array(color img),
    :param coordinates: tuple or list(iterable), ((x y, x, y), (x y, x, y) .. (x y, x, y))
    내부적으로는 x y x y 좌표계를 가지고 있어야 함
    :return: img: ndarray, 3d array, shape = (H, W, CH)
    """
    for coord in coordinates:
        img = draw_rectangle(img, coord, color)
    return np.array(img)


def images_with_rectangles(imgs, bboxes_bucket, color=(255, 0, 0)):
    """
    여러개의 이미지에 여러개의 bouding box 을 그리는 알고리즘.

    :param imgs: ndarray , 4d array, N H W C 구조를 가지고 있음
    :param bboxes_bucket:tuple or list(iterable),
        (
        (x y, x, y), (x y, x, y) .. (x y, x, y),
        (x y, x, y), (x y, x, y) .. (x y, x, y),
                    ...
        (x y, x, y), (x y, x, y) .. (x y, x, y),
        )
    :return: list, 4d array,  N H W C 구조를 가지고 있음
    """
    boxed_imgs = []
    for img, bboxes in zip(imgs, bboxes_bucket):
        # if gray image
        if img.shape[-1] == 1:
            img = np.squeeze(img)
        # draw bbox img
        bboxes_img = draw_rectangles(img, bboxes, color)
        boxed_imgs.append(bboxes_img)
    return boxed_imgs


def filter_xml_paths(paths):
    """
    Description:
        입력 경로중 '.gif|.jpg|.jpeg|.tiff|.png' 해당 확장자를 가진 파일만 반환합니다.
    Args:
    :param: paths, [str, str, str ... str]
    :return: list, [str, str, str ... str]
    """
    regex = re.compile("(.*)(\w+)(.xml)")
    xml_paths = []
    for path in paths:
        if regex.search(path):
            xml_paths.append(path)
    return xml_paths


def lxywh2cxywh(xywh):
    """
    Description:
    lx ly w h 좌표를 cx cy w h 좌표로 변환합니다
    :param xywh: shape (..., 4), 2차원 이상의 array 가 들어와야 함
    :return: xywh shape(... , 4)
    """
    w = xywh[..., 2:3]
    h = xywh[..., 3:4]
    cx = xywh[..., 0:1] + w * 0.5
    cy = xywh[..., 1:2] + h * 0.5
    xywh = np.concatenate([cx, cy, w, h], axis=1)
    return xywh


def xy2lonlat(x, y, raster_file):
    """
    xy 픽셀좌표계를 절대 좌표계로 변경
    Parameters
    ----------
    x int
    y int
    raster_file
        ex)
            file_path = '../pix4d_transparent_mosaic_group1.tif'  # 실제 파일 경로로 대체해주세요
            raster_file = rasterio.open(file_path)
    Returns
    -------

    """

    def transform_coordinates(crs1, crs2, x, y):
        transformer = Transformer.from_proj(crs1, crs2, always_xy=True)
        return transformer.transform(x, y)

    # TIF 파일 불러오기
    # 좌표 변환을 위해 필요한 CRS 가져오기
    src_crs = raster_file.crs
    dst_crs = 'EPSG:4326'  # 예시 값, 원하는 좌표 참조 시스템으로 대체해야 합니다.

    # c, f 값을 경도,  변환
    lon, lat = transform_coordinates(src_crs, dst_crs, x, y)
    return lon, lat


def lonlat2xy(lon, lat, raster_file):
    # 좌표 변환을 위한 함수
    def transform_coordinates(crs1, crs2, x, y):
        transformer = Transformer.from_proj(crs1, crs2, always_xy=True)
        return transformer.transform(x, y)

    # 좌표 변환을 위해 필요한 CRS 가져오기
    src_crs = 'EPSG:4326'  # 예시 값, 원하는 좌표 참조 시스템으로 대체해야 합니다.
    dst_crs = raster_file.crs

    # c, f 값을 위도, 경도로 변환
    x, y = transform_coordinates(src_crs, dst_crs, lon, lat)
    return x, y


def xyxy2lonlat(xyxy, raster_file):
    """
    xy xy  좌표를 lon lat lon lat 로 변경해 반환합니다.
    Parameters
    ----------
    xyxy : tuple or list
    (x1, y1, x2, y2)

    Returns : tuple or list
    (lon, lat, lon, lat)
    -------

    """
    lon1, lat1 = xy2lonlat(xyxy[0], xyxy[1], raster_file)
    lon2, lat2 = xy2lonlat(xyxy[2], xyxy[3], raster_file)
    return lon1, lat1, lon2, lat2


def lonlat2xyxy(lon_lat, raster_file):
    """
    lon lat lon lat 좌표를 xy xy 로 변경해 반환합니다.
    Parameters
    ----------
    lon_lat : tuple or list
    (lon, lat, lon, lat)

    Returns :tuple or list
    (x1, y1, x2, y2)
    -------

    """

    x1, y1 = lonlat2xy(lon_lat[0], lon_lat[1], raster_file)
    x2, y2 = lonlat2xy(lon_lat[2], lon_lat[3], raster_file)
    return x1, y1, x2, y2


def xy2lonlat(x, y, raster_file):
    """
    raster_file 내 정의된 기본 좌표계 값을 EPSG: 4326(경도 위도) 로 변경한다.
    Parameters
    ----------
    x float : raster_file 내 정의된 기본 좌표계 값
    y float : raster_file 내 정의된 기본 좌표계 값
    raster_file

    Returns
    -------

    """

    def transform_coordinates(crs1, crs2, x, y):
        transformer = Transformer.from_proj(crs1, crs2, always_xy=True)
        return transformer.transform(x, y)

    # TIF 파일 불러오기
    # 좌표 변환을 위해 필요한 CRS 가져오기
    src_crs = raster_file.crs
    dst_crs = 'EPSG:4326'  # 예시 값, 원하는 좌표 참조 시스템으로 대체해야 합니다.

    # x, y 값을 경도, 위도로 변환
    lon, lat = transform_coordinates(src_crs, dst_crs, x, y)
    return lon, lat


def xy2lonlat_vectorize(xs, ys, raster_file):
    xs = xs.reshape(-1)
    ys = ys.reshape(-1)
    vfunc = np.vectorize(xy2lonlat)
    lons, lats = vfunc(xs, ys, raster_file)
    return lons, lats


def lonlat2xy(lon, lat, raster_file):
    # 좌표 변환을 위한 함수
    def transform_coordinates(crs1, crs2, x, y):
        transformer = Transformer.from_proj(crs1, crs2, always_xy=True)
        return transformer.transform(x, y)

    # 좌표 변환을 위해 필요한 CRS 가져오기
    src_crs = 'EPSG:4326'  # 예시 값, 원하는 좌표 참조 시스템으로 대체해야 합니다.
    dst_crs = raster_file.crs

    # c, f 값을 위도, 경도로 변환
    x, y = transform_coordinates(src_crs, dst_crs, lon, lat)
    return x, y


def lonlat2xy_vectorize(lons, lats, raster_file):
    lons = lons.reshape(-1)
    lats = lats.reshape(-1)
    print(lons.shape, lats.shape)
    vfunc = np.vectorize(lonlat2xy)
    xs, ys = vfunc(lons, lats, raster_file)
    return xs.astype(int), ys.astype(int)


def search_mininum_bbox(polygon_coords):
    """
        polygon 좌표를 포함하는 rotated bounding box 좌표를 생성해 반환

    :param polygon_coords: [(x1, y1), (x_1, y_1) ... (x_n`, y_n`)]
    :return:
        min_bounding_rect_coords ([x1, y1], [x2 ,y1], [x2, y2], [x1, y2])
    """
    # 폴리곤 좌표를 나타내는 리스트를 만듭니다.
    # polygon_coords = [(0, 0), (2, 0), (2, 2), (0, 2)]

    # Shapely Polygon 객체를 생성합니다.
    polygon = Polygon(polygon_coords)

    # 폴리곤을 포함하는 최소한의 직사각형을 얻습니다.
    min_bounding_rect = polygon.minimum_rotated_rectangle

    # 최소한의 직사각형의 좌표를 얻습니다.
    min_bounding_rect_coords = list(min_bounding_rect.exterior.coords)

    # print("최소한의 직사각형 좌표:", min_bounding_rect_coords)
    # [[x1, y1] , [x2 ,y1], [x2, y2], [x1, y2], [x1, y1]] 이기 때문에 마지막 좌표를 제거(마지막 좌표가 처음 좌표와 같다)
    return min_bounding_rect_coords[:4]


def calculate_rotated_rect_angle(rotated_rect_coords):
    """
    직사각형을 이루는 두 대각선의 방향을 계산합니다.

    :param list or ndarray rotated_rect_coords:
        ([x1, y1], [x2 ,y1], [x2, y2], [x1, y2])
    :int return:
        rotation_angle
    """
    #
    # 첫 번째 대각선의 방향
    p1, p2, p3, p4 = rotated_rect_coords[:4]
    angle1 = math.atan2(p2[1] - p1[1], p2[0] - p1[0])

    # 두 번째 대각선의 방향
    angle2 = math.atan2(p4[1] - p3[1], p4[0] - p3[0])

    # 두 대각선 중에서 더 작은 각도를 선택합니다.
    rotation_angle = min(abs(angle1), abs(angle2)) * (180.0 / math.pi)

    return rotation_angle


def rotate_images(images, angle):
    """
    이미지를 지정된 각도로 회전합니다. 회전시 이미지내 객체를 잘리지 않게 보정합니다.

    :param ndarray image:
    :param float angle:
    :return:
    """
    import imgaug.augmenters as iaa
    seq = iaa.Sequential([iaa.Affine(rotate=angle, fit_output=True)])
    images = np.array(images)
    rotated_image = seq(images=images)
    return rotated_image


def remove_zero_padding(image):
    """
    이미지내 불 필요한 padding 을 제거합니다.
    :param image:
        shape=(H, W, CH)
    :return ndarray:
        shape=(H, W, CH)
    """
    # 이미지를 그레이스케일로 변환
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 그레이스케일 이미지에서 0이 아닌 픽셀의 좌표를 찾습니다.
    non_zero_coords = np.column_stack(np.where(gray_image > 0))

    # 마스크 생성(값이 0이 아닌 pixel 을 1로 함)
    mask = np.where(gray_image > 0, 1, 0)
    l_y = np.argwhere(mask)[:, 0].min()
    r_y = np.argwhere(mask)[:, 0].max()
    l_x = np.argwhere(mask)[:, 1].min()
    r_x = np.argwhere(mask)[:, 1].max()

    # 원본 이미지에서 직사각형 부분만을 가져옵니다.
    cropped_img = image[l_y: r_y, l_x: r_x]

    return cropped_img


def clahe(image):
    gray_img = np.array(Image.fromarray(image).convert('L'))
    assert len(gray_img.shape) == 2, '2차원으로 되어야 합니다. {} '.format(gray_img.shape)

    # contrast limit가 2이고 title의 size는 8X8
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray_img)

    return clahe_img


def add_zero_padding(image, pad_h, pad_w):
    # 추출된 이미지 양 끝에 주변에 zero padding 을 붙입니다.
    pad_h, pad_w = (np.array(image.shape) / 4).astype(int)[:2]
    if image.ndim == 3:
        pad_width = [(pad_h, pad_w), (pad_h, pad_w), (0, 0)]
    elif image.ndim == 2:
        pad_width = [(pad_h, pad_w), (pad_h, pad_w)]
    else:
        print('지원하지 않는 이미지 입니다. 3차원 또는 2차원 이미지만 지원합니다.')
        raise NotImplementedError
    padded_image = np.pad(image, pad_width)
    return padded_image


def draw_polygons(img, coords, **kwargs):
    """
    이미지에 다각형을 그리는 함수

    Parameters:
    - img: numpy.ndarray
        그림을 그릴 대상 이미지 배열
    - coords: list of int
        다각형 좌표를 담은 리스트. 각 좌표는 (x, y) 형태로 번갈아가며 포함되어야 함.
    - **kwargs: 키워드 인자
        OpenCV의 cv2.polylines 함수에 전달할 추가적인 인자들

    Returns:
    - polylined_img: numpy.ndarray
        다각형이 그려진 이미지 배열

    Example:
    ```python
    import cv2

    # 이미지 생성
    img = np.zeros((500, 500, 3), dtype=np.uint8)

    # 다각형 좌표
    polygon_coords = [100, 100, 200, 50, 300, 100, 250, 200, 150, 200]

    # 다각형 그리기
    result_img = draw_polygons(img, polygon_coords, color=(0, 255, 0), thickness=5)

    # 결과 이미지 출력
    cv2.imshow('Polygon Image', result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    ```

    Notes:
    - 이 함수는 입력 이미지에 지정된 좌표를 사용하여 다각형을 그리고 반환합니다.
    - OpenCV의 cv2.polylines 함수를 내부적으로 호출하며, 그에 대한 추가 인자는 **kwargs를 통해 전달됩니다.
    """
    # 단일 polygon을 이미지에 그립니다.
    resized_coords = (np.array(coords).reshape(-1, 2)).astype(int)
    polylined_img = cv2.polylines(img, [resized_coords], isClosed=True, **kwargs)
    return polylined_img


def draw_ordered_coordinates(sample_img, coords, **kwargs):
    """
    이미지에 주어진 좌표를 순서대로 표기함
    """
    for ind, coord in enumerate(coords):
        coord = np.array(coord).astype(int)
        sample_img = cv2.putText(sample_img, str(ind), coord, **kwargs)
    return sample_img


def calculate_rectangle_lengths(rectangle):
    """
    회전된 직사각형의 두 변의 길이를 계산하는 함수

    Parameters:
    - rectangle: list of tuples
        회전된 직사각형의 네 꼭짓점 좌표를 담은 리스트. 각 좌표는 (x, y) 형태로 표현됨.

    Returns:
    - side_lengths: tuple
        두 변의 길이를 나타내는 튜플 (long, short), 여기서 long은 더 긴 변의 길이, short는 더 짧은 변의 길이

    Raises:
    - AssertionError: 입력된 좌표가 직사각형을 형성하지 않는 경우 에러를 발생시킴
    """
    # 각 꼭짓점을 연결하는 선분의 길이를 계산
    side1 = int(np.linalg.norm(np.array(rectangle[1]) - np.array(rectangle[0])))
    side2 = int(np.linalg.norm(np.array(rectangle[2]) - np.array(rectangle[1])))
    side3 = int(np.linalg.norm(np.array(rectangle[3]) - np.array(rectangle[2])))
    side4 = int(np.linalg.norm(np.array(rectangle[0]) - np.array(rectangle[3])))

    # 직사각형은 4면 중 2변은 같은 길이를 가져야 합니다.
    assert (side1 == side3) and (side2 == side4), "입력된 좌표가 직사각형을 형성하지 않습니다. {},{},{},{}".format(side1,
                                                                                                side2,
                                                                                                side3,
                                                                                                side4)

    # 더 긴 변과 더 짧은 변을 계산
    long = np.maximum(side1, side2)
    short = np.minimum(side1, side2)

    side_lengths = (long, short)
    return side_lengths
