"""
输出格式：
id x1 x2 x3 x4 conf(归一化)
"""

import cv2
import os
from src.detect.det import DetYoloV8
from src.catch.catch import Catch
from src.fai.fai import Index


def main():
    # 初始化所有
    det = DetYoloV8(det_model_path)
    cat = Catch()  # 内置的模型
    index = Index(index_data, idmap_data)

    # 循环预测test
    for root, dirs, files in os.walk(test_path):
        for fe in files:
            if fe.lower().endswith(('.png', '.jpg')):
                file_path = os.path.join(root, fe)
                file_name = fe.split(".")[0]
                save_txt_path = os.path.join(res_path, file_name+'.txt')
                det_res = det.run(file_path)
                img = cv2.imread(file_path)
                with open(save_txt_path, 'w') as write:
                    for d in det_res:
                        y1 = int(d["y1"])
                        x1 = int(d["x1"])
                        y2 = int(d["y2"])
                        x2 = int(d["x2"])
                        crop = img[y1:y2, x1:x2]
                        feature = cat.run(crop)
                        item = index.run(feature)[0]
                        # 获取归一化坐标
                        x_center = round(float(d["x_center"]),6)
                        y_center = round(float(d["y_center"]),6)
                        w = round(float(d["w"]),6)
                        h = round(float(d["h"]),6)
                        # 这里做个conf 转换  用distance -> conf 100-distance = conf, 如果conf>100 则认为结果无效。conf=0.0
                        class_id = item['class_name']
                        dis = float(item['distance'])
                        conf = (100-dis)/100 if dis<100 else 0.0
                        w_str = "{} {} {} {} {} {}\n ".format(class_id,x_center,y_center,w,h,round(conf,6))
                        print(w_str)
                        write.write(w_str)


if __name__ == "__main__":
    det_model_path = r"F:\comptition\model\det.pt"
    index_path = r"F:\comptition\model\index.bin"
    idmap_path = r"F:\comptition\model\idmap.txt"
    with open(idmap_path, "rb") as r:
        idmap_data = r.read()
    with open(index_path, "rb") as r:
        index_data = r.read()
    test_path = r"E:\fish\comp\test\test\images"
    res_path = r"E:\fish\comp\test\test\labels"
    main()
