# face_identification
・画像からアイドルの顔検出/識別を行い、プロットした画像を作成  
・顔識別は[FaceNet](https://github.com/timesler/facenet-pytorch)を使用
![title](https://github.com/user-attachments/assets/aad75e6f-0831-4bc8-9574-7c04afe4e644)


## データ加工
#### 顔画像切り取り
```
python preprocess/cut_face.py
```
#### データ拡張（学習用）
```
python preprocess/data_aug.py
```


## 学習
以下を設定して実行
<details><summary>設定項目</summary>

・`DATANAME`    ：学習に使用するデータ  
・`LISTNAME`    ：メンバーのリスト  
・`AUG_DIR`     ：データ拡張を行ったデータ  
・`AUG_SIZE`    ：1枚の画像に対してデータ拡張で増やす枚数
・`DEVICE`      ：cuda or mps or cpu  

</details>
```
python train.py
```


## 推論
#### 実行
画像に含まれる人物の顔からメンバーの名前をプロット
```
python inference/img2names.py
```


## サンプル
#### 単一メンバー
![single_img](https://github.com/user-attachments/assets/9aa4becc-a0ac-4780-93a0-c54c3651d842)
#### 複数メンバー
![sample1_jp](https://github.com/user-attachments/assets/0fe74957-5d7d-47eb-8242-8ad3ddfbbea3)
![sample2_jp](https://github.com/user-attachments/assets/b3cfd422-daa9-4f92-bc13-d288ddb2695c)