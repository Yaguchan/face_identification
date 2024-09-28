# face_identification
・画像からアイドルの顔検出/識別を行い、プロットした画像を作成  
・顔識別は[FaceNet](https://github.com/timesler/facenet-pytorch)を使用
![title](https://github.com/user-attachments/assets/aad75e6f-0831-4bc8-9574-7c04afe4e644)

## 実行
## 環境構築
```
conda env create -f env.yaml
```

### データ加工
（Xで取得した）単一メンバーの画像から顔画像の切り取り
```
python preprocess/cut_face.py
```
データ拡張
```
python preprocess/data_aug.py
```

### 学習
以下を設定して、FaceNetをファインチューニング  
事前学習モデルは`weights/facenet`から利用可能
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

### 推論
以下を設定して、画像に含まれる人物の顔からメンバーの名前をプロット
<details><summary>設定項目</summary>

・`MODELPATH`   ：FaceNet model  
・`IMG_PATH`    ：プロットする画像  
・`MEMBER_LIST` ：メンバーのリスト  
・`MEMBER_ENJP` ：メンバーの名前の日本語/英語データ  
・`FONT_PATH`   ：使用するフォント  
・`FONT_SIZE`   ：使用するフォントサイズ（normal or large）  
・`DEVICE`      ：cuda or mps or cpu  
・`JP`          ：プロット（日本語 or 英語）

</details>

```
python inference/img2names.py
```


## サンプル
### 単一メンバー
![single_img](https://github.com/user-attachments/assets/9aa4becc-a0ac-4780-93a0-c54c3651d842)
### 複数メンバー
![sample1_jp](https://github.com/user-attachments/assets/0fe74957-5d7d-47eb-8242-8ad3ddfbbea3)
![sample2_jp](https://github.com/user-attachments/assets/b3cfd422-daa9-4f92-bc13-d288ddb2695c)