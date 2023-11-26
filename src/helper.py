import pandas as pd
import matplotlib.pyplot as plt
from utils import (get_names, paths2imgs, write_msg_in_polygon_center, search_mininum_bbox,
                   calculate_rotated_rect_angle, rotate_images, remove_zero_padding)
import numpy as np
from utils import path2img, extract_pixel_in_polygon
from PIL import Image


def write_index_in_abalone_center(sample_anno_df, **kwargs):
    # 특정 조건 내 전복 데이터에 해당하는 이미지를 불러와 로드 합니다.
    image_paths = sample_anno_df.image_path.value_counts().index
    image_names = get_names(image_paths, ext=True)
    images = paths2imgs(image_paths)

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
    for idx, row in masked_anno_df.iterrows():
        # 폴리곤 내 픽셀만 추출해 반환합니다.
        sample_img = path2img(row.image_path)
        sample_seg = row.segmentation
        sample_ply = extract_pixel_in_polygon(sample_img, sample_seg)

        # 장축이 가장 길어지도록 전복을 회전합니다.
        polygon_coords = np.array(row.segmentation).reshape(-1, 2)
        min_bounding_rect_coords = search_mininum_bbox(polygon_coords)
        rot_angle = calculate_rotated_rect_angle(min_bounding_rect_coords)
        rot_image = rotate_images([sample_ply], rot_angle)
        rot_image = remove_zero_padding(rot_image[0])

        ply_images.append(rot_image)
        ply_meta.append(int(row.meta_index))

    # meta index 기준으로 오름차순으로 정렬
    sorted_index = np.argsort(ply_meta)
    sorted_ply_meta = np.array(ply_meta)[sorted_index]
    sorted_ply_images = np.array(ply_images, dtype=object)[sorted_index]

    return sorted_ply_images, sorted_ply_meta


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


def basic_analysis(images, meta_index, show=True):
    """

    Args:
        :list images:
         [ndarray, ndarray ... ]
        :list meta_index:
         [1, 2, 3... ]

    Returns:
        list longs:
        list shorts
        list ratios
        DataFrame describe:
    """

    longs, shorts = get_abalone_long_short_from_images(images)
    longs = np.array(longs) / 34
    shorts = np.array(shorts) / 34
    if show:
        show_abalone_longshort(longs, shorts, meta_index)
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


def compare_jindo_wando(wando_images, wando_meta_index, jindo_images, jindo_meta_index):
    WANDO_COLOR = 'C0'
    JINDO_COLOR = 'C1'

    w_longs, w_shorts, w_ratios, w_describe = basic_analysis(wando_images, wando_meta_index)
    j_longs, j_shorts, j_ratios, j_describe = basic_analysis(jindo_images, jindo_meta_index)

    # long, short 시각화
    plt.scatter(w_longs, w_shorts, c=WANDO_COLOR, alpha=0.5, label='Wando')
    for idx, (long, short) in enumerate(zip(w_longs, w_shorts)):
        plt.annotate(wando_meta_index[idx], (long, short))

    plt.scatter(j_longs, j_shorts, c=JINDO_COLOR, alpha=0.5, label='Jindo')
    for idx, (long, short) in enumerate(zip(j_longs, j_shorts)):
        plt.annotate(jindo_meta_index[idx], (long, short))
    plt.legend()
    plt.show()
    # ratio 출력
    print('완도산 L/S 비율 : {} \n진도산 L/S 비율 : {}'.format(np.mean(w_ratios), np.mean( j_ratios)))

    # wando , jindo describe 비교 시각화
    target_index = [1, 3, 7]
    target_name = ['mean', 'min', 'max']
    x_coord = [0, 1, 2]
    print(pd.concat([w_describe, j_describe], axis=1))

    w_long_info = w_describe.iloc[target_index, 0]
    w_short_info = w_describe.iloc[target_index, 1]
    j_long_info = j_describe.iloc[target_index, 0]
    j_short_info = j_describe.iloc[target_index, 1]

    # 장축 비교 시각화
    width = 0.4
    plt.bar(x_coord, height=w_long_info.values, color=WANDO_COLOR, label='Wando/long', alpha=0.5, width=width)
    plt.bar(np.array(x_coord) + width, height=j_long_info, color=JINDO_COLOR, label='Jindo/long', alpha=0.5, width=width)
    plt.xticks(np.array(x_coord) + width/2, target_name)
    plt.title('Compare Wando with Jindo, long axis')
    plt.legend()
    plt.show()

    # 단축 비교 시각화
    plt.bar(x_coord, height=w_short_info.values, color=WANDO_COLOR, label='Wando/short', alpha=0.5, width=width)
    plt.bar(np.array(x_coord) + width, height=j_short_info, color=JINDO_COLOR, label='Jindo/short', alpha=0.5,
            width=width)
    plt.xticks(np.array(x_coord)+width/2, target_name)
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
    plt.bar(np.array(x_coord)+width, height=[j_r_mean, j_g_mean, j_b_mean], color=JINDO_COLOR, label='Jindo',
            alpha=0.5, width=width)
    plt.xticks(np.array(x_coord) + width/2, ['Red', 'Green', 'Blue'])
    plt.legend()
    plt.show()

    print('Wando : R:{} G:{} B{}'.format(w_r_mean, w_g_mean, w_b_mean))
    print('Jindo : R:{} G:{} B{}'.format(j_r_mean, j_g_mean, j_b_mean))
