# face_identification
・顔の切り出しは[YOLOv8-face](https://github.com/akanametov/yolo-face)を使用
・顔識別は[facenet-pytorch](https://github.com/timesler/facenet-pytorch)を使用

## データ加工(データ拡張用)
解像度を変更した画像を作成
```
python preprocess/data_resize.py
```
## 学習
学習に使用するデータとそのメンバーリストを与えて学習
```
python run.py
```
## 推論
### 実行
画像にメンバーの名前を割り当てる
```
python inference/img2names.py
```
### サンプル
data/sample/sample1_plot.jpg
![sample1_plot](https://github.com/user-attachments/assets/6a2e37c6-e4fb-43e4-9989-86794be744ec)