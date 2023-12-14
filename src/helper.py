import pandas as pd
import matplotlib.pyplot as plt
from utils import get_names, paths2imgs, write_msg_in_polygon_center, search_mininum_bbox, draw_rectangle
from utils import calculate_rotated_rect_angle, rotate_images, remove_zero_padding, add_zero_padding
from utils import draw_polygons, draw_ordered_coordinates
import numpy as np
from utils import path2img, extract_pixel_in_polygon, calculate_rectangle_lengths
from PIL import Image
import cv2


def write_index_in_abalone_center(sample_anno_df, **kwargs):
    # 특정 조건 내 전복 데이터에 해당하는 이미지를 불러와 로드 합니다.
    image_paths = sample_anno_df.image_path.value_counts().index  #
    image_names = get_names(image_paths, ext=True)  #
    images = paths2imgs(image_paths)  #

    # 각 이미지 내 위치하는 전복 껍질위에 DataFrame index 번호를 기입합니다.
    for idx, image_name in enumerate(image_names):
        tmp_df = sample_anno_df.loc[sample_anno_df.image_id == image_name]
        for df_index, row in tmp_df.iterrows():
            tmp_seg = np.array(row.segmentation).reshape(-1, 2).tolist()
            images[idx] = write_msg_in_polygon_center(images[idx], str(df_index), *tmp_seg, **kwargs)
    return images, image_names


def extract_albarone_polygon_images(anno_df, **kwargs):
    """
    anno_df 내 존재하는 전복 polygon 이미지들을 모두 추출해 반환합니다.

    :param anno_df: DataFrame
    :param kwargs:
        region: JINDO or WANDO
        entity: ADULT or BABY
        size_: BIG or SMALL
        side: f or b
    :return:
        list sorted_ply_images: 정렬된 ndarray 이미지들이 들어 있는 리스트
        list sorted_ply_meta: meta_index 리스트
    """

    # DataFrame 내 조건에 맞는 row를 추출
    mask = pd.Series([True] * len(anno_df), index=anno_df.index)  # mask 생성시 anno df 와 같은 index 을 가지도록 해야한다.
    if kwargs['region']:
        mask &= anno_df.region == kwargs['region']
    if kwargs['entity']:
        mask &= anno_df.entity == kwargs['entity']
    if kwargs['size_']:
        mask &= anno_df.size_ == kwargs['size_']
    if kwargs['side']:
        mask &= anno_df.side == kwargs['side']
    masked_anno_df = anno_df.loc[mask]

    # filter 된 DataFrame 내 모든 폴리곤 이미지 추출
    ply_images = []
    ply_meta = []
    ply_info = []
    for idx, row in masked_anno_df.iterrows():
        # 폴리곤 내 픽셀만 추출해 반환합니다.
        sample_img = path2img(row.image_path)
        sample_seg = row.segmentation
        sample_ply = extract_pixel_in_polygon(sample_img, sample_seg)

        # 장축이 가장 길어지도록 전복을 회전합니다.
        polygon_coords = np.array(row.segmentation).reshape(-1, 2)
        min_bounding_rect_coords = search_mininum_bbox(polygon_coords)
        rot_angle = calculate_rotated_rect_angle(min_bounding_rect_coords)

        # debugging
        # 이미지 내 전복 객체에 그려진 rotated bbox 을 확인합니다.
        debug = False
        if debug:
            draw_img = draw_polygons(sample_img, min_bounding_rect_coords, color=(0, 255, 0), thickness=5)
            draw_img = draw_ordered_coordinates(draw_img, min_bounding_rect_coords,
                                                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                                fontScale=3,
                                                color=(0, 0, 255),
                                                thickness=10)
            plt.imshow(draw_img)
            plt.show()


        # 이미지를 계산된 각도로 회전하여 평평한 상태로 변환합니다.
        rot_image = rotate_images([sample_ply], rot_angle)
        rot_image = remove_zero_padding(rot_image[0])

        # 객체를 포함하고 있는 직사각형의 장축과 단축을 계산합니다.
        long, short = calculate_rectangle_lengths(min_bounding_rect_coords)

        # python 변수에 이미지와 meta 번호 , 장축, 단축 길이를 저장합니다.
        ply_images.append(rot_image)
        ply_meta.append(int(row.meta_index))
        ply_info.append((int(row.meta_index), long, short))

    # meta index 기준으로 오름차순으로 정렬
    sorted_index = np.argsort(ply_meta)
    sorted_ply_info = np.array(ply_info)[sorted_index]
    sorted_ply_images = np.array(ply_images, dtype=object)[sorted_index]

    return sorted_ply_images, sorted_ply_info


def get_abalone_long_short_from_images(images):
    """
    이미지들의 height , width 을 얻어옵니다.
    :param images:
    :list return:
    """
    longs = []
    shorts = []
    for img in images:
        short_axis = np.min(img.shape[:2])
        long_axis = np.max(img.shape[:2])
        longs.append(long_axis)
        shorts.append(short_axis)
    return longs, shorts


def show_abalone_longshort(longs, shorts, meta_index):
    import matplotlib.pyplot as plt
    plt.scatter(longs, shorts)
    for idx, (long, short) in enumerate(zip(longs, shorts)):
        plt.annotate(meta_index[idx], (long, short))


def basic_analysis(meta_index, show=True):
    """

    Args:
        :list images:
         [ndarray, ndarray ... ]
        :list meta_index: [(index, long, short),(index, long, short) ... (index, long, short)]
         [(), 2, 3... ]

    Returns:
        list longs:
        list shorts
        list ratios
        DataFrame describe:
    """
    # 1cm 당 pixel 개 수
    PIXEL_PER_CM = 34

    # get abalone long short
    index = np.array(meta_index)[:, 0]
    longs = np.array(meta_index)[:, 1]
    shorts = np.array(meta_index)[:, 2]

    # 1cm = 32pixel, pixel 길이를 cm 로 변환합니다.
    longs = np.array(longs) / 34
    shorts = np.array(shorts) / 34

    # 장축, 단축 분석 비교 분석 결과를 시각화 합니다.
    if show:
        show_abalone_longshort(longs, shorts, index)
    ratios = longs / shorts

    # 기초량 분석
    describe = pd.concat([pd.DataFrame(longs).describe(),
                          pd.DataFrame(shorts).describe(),
                          pd.DataFrame(ratios).describe()], axis=1)
    describe.columns = ['long(cm)', 'short(cm)', 'ratios']
    return longs, shorts, ratios, describe


def rgb_analysis(images):
    # RGB 분석

    rs = []
    gs = []
    bs = []
    for img in images:
        img = np.array(Image.fromarray(img).convert('RGB'))
        rs.extend(img[0].reshape(-1).tolist())
        gs.extend(img[1].reshape(-1).tolist())
        bs.extend(img[2].reshape(-1).tolist())

    # show RGB histogram
    fig, axes = plt.subplots(1, 3)
    fig.set_size_inches(20, 5)
    axes[0].hist(rs, bins=np.arange(1, 256))
    axes[0].set_title('Red channel')
    axes[1].hist(gs, bins=np.arange(1, 256))
    axes[1].set_title('Green channel')
    axes[2].hist(bs, bins=np.arange(1, 256))
    axes[2].set_title('Blue channel')
    plt.show()

    # RGB MEAN
    r_mean, g_mean, b_mean = np.mean(rs), np.mean(gs), np.mean(bs)
    print(r_mean, g_mean, b_mean)



def compare_jindo_wando(wando_images, wando_meta_info, jindo_images, jindo_meta_info):
    WANDO_COLOR = 'C0'
    JINDO_COLOR = 'C1'

    s_longs, s_shorts, s_ratios, s_describe = basic_analysis(wando_meta_info)
    b_longs, b_shorts, b_ratios, b_describe = basic_analysis(jindo_meta_info)

    # long, short 시각화
    plt.scatter(s_longs, s_shorts, c=WANDO_COLOR, alpha=0.5, label='Wando')
    for idx, (long, short) in enumerate(zip(s_longs, s_shorts)):
        plt.annotate(wando_meta_info[idx][0], (long, short))

    plt.scatter(b_longs, b_shorts, c=JINDO_COLOR, alpha=0.5, label='Jindo')
    for idx, (long, short) in enumerate(zip(b_longs, b_shorts)):
        plt.annotate(jindo_meta_info[idx][0], (long, short))
    plt.legend()
    plt.show()
    # ratio 출력
    print('완도산 L/S 비율 : {} \n진도산 L/S 비율 : {}'.format(np.mean(s_ratios), np.mean(b_ratios)))

    # wando , jindo describe 비교 시각화
    target_index = [1, 3, 7]
    target_name = ['mean', 'min', 'max']
    x_coord = [0, 1, 2]
    print(pd.concat([s_describe, b_describe], axis=1))

    w_long_info = s_describe.iloc[target_index, 0]
    w_short_info = s_describe.iloc[target_index, 1]
    j_long_info = b_describe.iloc[target_index, 0]
    j_short_info = b_describe.iloc[target_index, 1]

    # 장축 비교 시각화
    width = 0.4
    plt.bar(x_coord, height=w_long_info.values, color=WANDO_COLOR, label='Wando/long', alpha=0.5, width=width)
    plt.bar(np.array(x_coord) + width, height=j_long_info, color=JINDO_COLOR, label='Jindo/long', alpha=0.5,
            width=width)
    plt.xticks(np.array(x_coord) + width / 2, target_name)
    plt.title('Compare Wando with Jindo, long axis')
    plt.legend()
    plt.show()

    # 단축 비교 시각화
    plt.bar(x_coord, height=w_short_info.values, color=WANDO_COLOR, label='Wando/short', alpha=0.5, width=width)
    plt.bar(np.array(x_coord) + width, height=j_short_info, color=JINDO_COLOR, label='Jindo/short', alpha=0.5,
            width=width)
    plt.xticks(np.array(x_coord) + width / 2, target_name)
    plt.title('Compare Wando with Jindo, short axis')
    plt.legend()
    plt.show()

    # RGB 분석
    fig, axes = plt.subplots(1, 3)
    fig.set_size_inches(20, 5)

    wando_rs, jindo_rs = [], []
    wando_gs, jindo_gs = [], []
    wando_bs, jindo_bs = [], []

    for img in wando_images:
        img = np.array(Image.fromarray(img).convert('RGB'))
        wando_rs.extend(img[..., 0].reshape(-1).tolist())
        wando_gs.extend(img[..., 1].reshape(-1).tolist())
        wando_bs.extend(img[..., 2].reshape(-1).tolist())

    for img in jindo_images:
        img = np.array(Image.fromarray(img).convert('RGB'))
        jindo_rs.extend(img[..., 0].reshape(-1).tolist())
        jindo_gs.extend(img[..., 1].reshape(-1).tolist())
        jindo_bs.extend(img[..., 2].reshape(-1).tolist())

    # show RGB histogram
    axes[0].hist(wando_rs, bins=np.arange(1, 256), color=WANDO_COLOR, label='wando', alpha=0.5)
    axes[1].hist(wando_gs, bins=np.arange(1, 256), color=WANDO_COLOR, label='wando', alpha=0.5)
    axes[2].hist(wando_bs, bins=np.arange(1, 256), color=WANDO_COLOR, label='wando', alpha=0.5)

    axes[0].hist(jindo_rs, bins=np.arange(1, 256), color=JINDO_COLOR, label='jindo', alpha=0.5)
    axes[1].hist(jindo_gs, bins=np.arange(1, 256), color=JINDO_COLOR, label='jindo', alpha=0.5)
    axes[2].hist(jindo_bs, bins=np.arange(1, 256), color=JINDO_COLOR, label='jindo', alpha=0.5)

    axes[0].set_title('Red channel')
    axes[0].legend()
    axes[1].set_title('Green channel')
    axes[1].legend()
    axes[2].set_title('Blue channel')
    axes[2].legend()
    plt.show()

    # RGB MEAN
    x_coord = [0, 1, 2]
    w_r_mean, w_g_mean, w_b_mean = np.mean(wando_rs), np.mean(wando_gs), np.mean(wando_bs)
    j_r_mean, j_g_mean, j_b_mean = np.mean(jindo_rs), np.mean(jindo_gs), np.mean(jindo_bs)
    plt.bar(np.array(x_coord), height=[w_r_mean, w_g_mean, w_b_mean], color=WANDO_COLOR, label='Wando', alpha=0.5,
            width=width)
    plt.bar(np.array(x_coord) + width, height=[j_r_mean, j_g_mean, j_b_mean], color=JINDO_COLOR, label='Jindo',
            alpha=0.5, width=width)
    plt.xticks(np.array(x_coord) + width / 2, ['Red', 'Green', 'Blue'])
    plt.legend()
    plt.show()

    print('Wando : R:{} G:{} B{}'.format(w_r_mean, w_g_mean, w_b_mean))
    print('Jindo : R:{} G:{} B{}'.format(j_r_mean, j_g_mean, j_b_mean))


def compare_small_big(big_images, big_meta_info, small_images, small_meta_info):
    big_COLOR = 'C0'
    small_COLOR = 'C1'

    s_longs, s_shorts, s_ratios, s_describe = basic_analysis(big_meta_info)
    b_longs, b_shorts, b_ratios, b_describe = basic_analysis(small_meta_info)

    # long, short 시각화
    plt.scatter(s_longs, s_shorts, c=big_COLOR, alpha=0.5, label='big')
    for idx, (long, short) in enumerate(zip(s_longs, s_shorts)):
        plt.annotate(big_meta_info[idx][0], (long, short))

    plt.scatter(b_longs, b_shorts, c=small_COLOR, alpha=0.5, label='small')
    for idx, (long, short) in enumerate(zip(b_longs, b_shorts)):
        plt.annotate(small_meta_info[idx][0], (long, short))
    plt.legend()
    plt.show()
    # ratio 출력
    print('BIG L/S 비율 : {} \nSMALL L/S 비율 : {}'.format(np.mean(s_ratios), np.mean(b_ratios)))

    # big , small describe 비교 시각화
    target_index = [1, 3, 7]
    target_name = ['mean', 'min', 'max']
    x_coord = [0, 1, 2]
    print(pd.concat([s_describe, b_describe], axis=1))

    w_long_info = s_describe.iloc[target_index, 0]
    w_short_info = s_describe.iloc[target_index, 1]
    j_long_info = b_describe.iloc[target_index, 0]
    j_short_info = b_describe.iloc[target_index, 1]

    # 장축 비교 시각화
    width = 0.4
    plt.bar(x_coord, height=w_long_info.values, color=big_COLOR, label='big/long', alpha=0.5, width=width)
    plt.bar(np.array(x_coord) + width, height=j_long_info, color=small_COLOR, label='small/long', alpha=0.5,
            width=width)
    plt.xticks(np.array(x_coord) + width / 2, target_name)
    plt.title('Compare big with small, long axis')
    plt.legend()
    plt.show()

    # 단축 비교 시각화
    plt.bar(x_coord, height=w_short_info.values, color=big_COLOR, label='big/short', alpha=0.5, width=width)
    plt.bar(np.array(x_coord) + width, height=j_short_info, color=small_COLOR, label='small/short', alpha=0.5,
            width=width)
    plt.xticks(np.array(x_coord) + width / 2, target_name)
    plt.title('Compare big with small, short axis')
    plt.legend()
    plt.show()

    # RGB 분석
    fig, axes = plt.subplots(1, 3)
    fig.set_size_inches(20, 5)

    big_rs, small_rs = [], []
    big_gs, small_gs = [], []
    big_bs, small_bs = [], []

    for img in big_images:
        img = np.array(Image.fromarray(img).convert('RGB'))
        big_rs.extend(img[..., 0].reshape(-1).tolist())
        big_gs.extend(img[..., 1].reshape(-1).tolist())
        big_bs.extend(img[..., 2].reshape(-1).tolist())

    for img in small_images:
        img = np.array(Image.fromarray(img).convert('RGB'))
        small_rs.extend(img[..., 0].reshape(-1).tolist())
        small_gs.extend(img[..., 1].reshape(-1).tolist())
        small_bs.extend(img[..., 2].reshape(-1).tolist())

    # show RGB histogram
    axes[0].hist(big_rs, bins=np.arange(1, 256), color=big_COLOR, label='big', alpha=0.5)
    axes[1].hist(big_gs, bins=np.arange(1, 256), color=big_COLOR, label='big', alpha=0.5)
    axes[2].hist(big_bs, bins=np.arange(1, 256), color=big_COLOR, label='big', alpha=0.5)

    axes[0].hist(small_rs, bins=np.arange(1, 256), color=small_COLOR, label='small', alpha=0.5)
    axes[1].hist(small_gs, bins=np.arange(1, 256), color=small_COLOR, label='small', alpha=0.5)
    axes[2].hist(small_bs, bins=np.arange(1, 256), color=small_COLOR, label='small', alpha=0.5)

    axes[0].set_title('Red channel')
    axes[0].legend()
    axes[1].set_title('Green channel')
    axes[1].legend()
    axes[2].set_title('Blue channel')
    axes[2].legend()
    plt.show()

    # RGB MEAN
    x_coord = [0, 1, 2]
    w_r_mean, w_g_mean, w_b_mean = np.mean(big_rs), np.mean(big_gs), np.mean(big_bs)
    j_r_mean, j_g_mean, j_b_mean = np.mean(small_rs), np.mean(small_gs), np.mean(small_bs)
    plt.bar(np.array(x_coord), height=[w_r_mean, w_g_mean, w_b_mean], color=big_COLOR, label='big', alpha=0.5,
            width=width)
    plt.bar(np.array(x_coord) + width, height=[j_r_mean, j_g_mean, j_b_mean], color=small_COLOR, label='small',
            alpha=0.5, width=width)
    plt.xticks(np.array(x_coord) + width / 2, ['Red', 'Green', 'Blue'])
    plt.legend()
    plt.show()

    print('big : R:{} G:{} B{}'.format(w_r_mean, w_g_mean, w_b_mean))
    print('small : R:{} G:{} B{}'.format(j_r_mean, j_g_mean, j_b_mean))
