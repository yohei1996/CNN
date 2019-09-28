#入力サイズ
Height = 128
Width = 128

#クラスラベル
Class_label = [
    'OK',
    'NG'
]

#クラス数
Class_num = len(Class_label)

#学習データのパス
Train_dirs = [
    'C:\\Users\\nishitsuji\\Documents\\myfile\\python_tensorflow\\dataset\\train\\OK',
    'C:\\Users\\nishitsuji\\Documents\\myfile\\python_tensorflow\\dataset\\train\\NG'
]

#テストデータのパス
Test_dirs = [
    'C:\\Users\\nishitsuji\\Documents\\myfile\\python_tensorflow\\dataset\\test\\OK',
    'C:\\Users\\nishitsuji\\Documents\\myfile\\python_tensorflow\\dataset\\test\\NG'
]

#ミニバッチ
batch_size = 50

#データ拡張(data_loader.pyで使用)
Horizontal_flip = False
Vertical_flip = False