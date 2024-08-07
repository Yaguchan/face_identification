# face_identification
・顔の切り出しは[YOLOv8-face](https://github.com/akanametov/yolo-face)を使用  
・顔識別は[facenet-pytorch](https://github.com/timesler/facenet-pytorch)を使用

## データ加工
### 解像度の変更
```
python preprocess/data_resize.py
```
### データ拡張（学習用）
```
python preprocess/data_aug.py
```


## 学習
学習に使用するデータとそのメンバーリストを与えて学習
```
python train.py
```

## 推論
### 実行
画像に含まれる人物の顔からメンバーの名前をプロット
```
python inference/img2names.py
```
## サンプル
### 単一
![sample3_jp](https://github.com/user-attachments/assets/b209dc8d-e030-4f36-9d5e-f778b7ab2343)
![sample4_jp](https://github.com/user-attachments/assets/e7a7aecf-3386-4928-b016-31055f1eb403)
![sample5_jp](https://github.com/user-attachments/assets/7fa16cf7-3fe3-43c7-aaae-4a5a2f9b4937)
### 複数
![sample1_jp](https://github.com/user-attachments/assets/0fe74957-5d7d-47eb-8242-8ad3ddfbbea3)
![sample2_jp](https://github.com/user-attachments/assets/b3cfd422-daa9-4f92-bc13-d288ddb2695c)