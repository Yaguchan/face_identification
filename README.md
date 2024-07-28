# face_identification
・顔の切り出しは[YOLOv8-face](https://github.com/akanametov/yolo-face)を使用  
・顔識別は[facenet-pytorch](https://github.com/timesler/facenet-pytorch)を使用

## データ加工（データ拡張用）
解像度を変更した画像を作成
```
python preprocess/data_resize.py
```
## 学習
学習に使用するデータとそのメンバーリストを与えて学習
```
python train.py
```
## 推論
### 実行
画像にメンバーの名前を割り当てる
```
python inference/img2names.py
```
### サンプル
![sample1_jp](https://github.com/user-attachments/assets/0fe74957-5d7d-47eb-8242-8ad3ddfbbea3)
![sample2_jp](https://github.com/user-attachments/assets/b3cfd422-daa9-4f92-bc13-d288ddb2695c)