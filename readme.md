# Abalone  분석 

## 목적 

전복 껍질 이미지 데이터를 분석한다.

## 데이터 

- 데이터는 크게 3가지로 나뉘어진다. (자세한 내용은 wiki 참조)

  - 메타 데이터 : 연구소에서 직접 전복 껍질의 크기와 무게를 측정한 데이터
  - 이미지 데이터
    - 원본 이미지 : AI Scanner 로 촬영한 이미지
      - 원본 이미지 내 들어 있는 전복 객체는 모두 앞면 또는 후면으로만 되어 있다.
    - 전복 껍질 이미지: 이미지 내 전복 껍질만을 추출한 이미지
  - 라벨 데이터 : 전복 껍질을 포함하는 가장 작은 다각형 라벨 데이터

- 완도 / 진도 산 껍질 데이터가 있으며 데이터 종류는 크게 8가지로 나뉘어진다.

  - Adult / Baby 

  - big / small 

  - 진도 / 완도

    

## 개발 환경

```shell
python 3.9
pip install -r requirements.txt
```

- ⚠️ `pycharm` 에서 개발 시 `abalone-analysis/src` 경로에서 시작해야 한다. (그렇지 않으면 jupyter script 실행 시 경로 에러 발생)



## 데이터 전처리

> 라벨링 정보와 이미지 정보 그리고 메타 정보를 하나의 DataFrame 으로 취합합니다. 

- 전처리 단계 

  |      | 파일 이름                                                    | 산출물                        |
  | ---- | ------------------------------------------------------------ | ----------------------------- |
  | 1    | [1.전복 정보 데이터프레임화.ipynb](./src/1.전복 정보 데이터프레임화.ipynb) | ./datasets/anno.df            |
  | 2    | [2.DataFrame(index)_meta(index)번호매칭.ipynb](./src/2.DataFrame(index)_meta(index)번호매칭.ipynb) |                               |
  | 3    | [3.모든 폴리곤 추출(완도).ipynb](./src/3.모든 폴리곤 추출(완도).ipynb) | ./datasets/wando_add_meta.csv |
  | 4    | [4.모든 폴리곤 추출(진도).ipynb](./src/4.모든 폴리곤 추출(진도).ipynb) | ./datasets/jindo_add_meta.csv |

## 분석

> 데이터 전처리 단계를 통해 생성된 DataFrame 을 분석합니다.
> 해당 스크립트를 수행하기 위해서는 `./datasets/wando_add_meta.csv`, `./datasets/jindo_add_meta.csv` 가 존재해야 합니다

- 분석 단계
  - [5. 데이터분석.ipynb](src/5. 데이터분석.ipynb) 