import unittest
import cv2
import os
from src.detect.det import DetYoloV8
from src.catch.catch import Catch
from src.fai.fai import Index


class TestComp(unittest.TestCase):
    def setUp(self) -> None:
        self.det = DetYoloV8("../model/det.pt")
        self.cat = None
        self.det_res = None
        self.cat_res = []
        self.img_path = r"E:\fish\comp\origin\train\images\2022-06-10-13-58-08-44810_2048-1536.png"
        self.small_path = r"F:\comptition\adjust_class\adjust_class\class_19\2025-04-15-12-57-24-76494_2048-1536_9.png"
        self.small_path_2 = r"F:\comptition\adjust_class\adjust_class\class_19\2025-04-15-12-57-24-76494_2048-1536_24.png"
        self.index_path = "../model/index.bin"
        self.idmap_path = "../model/idmap.txt"
        self.idmap_data = None
        self.index_data = None
        self.fai = None

    def test_init_catch(self):
        # 测试特征提取器初始化
        self.cat = Catch()

    def test_init_fai(self):
        # 测试 检索库初始化
        with open(self.idmap_path, "rb") as r:
            self.idmap_data = r.read()
        with open(self.index_path, "rb") as r:
            self.index_data = r.read()
        self.fai = Index(self.index_data, self.idmap_data)

    def test_det(self):
        # 测试检测
        self.det_res = self.det.run(self.img_path)
        self.assertIsNotNone(self.det_res)

    def test_det_bulk_txt(self):
        # 批量检测 并 存txt
        img_base_path = r"E:\fish\comp\test\test\images"
        save_base = r"E:\fish\comp\test\test\submit"
        for r, d, fs in os.walk(img_base_path):
            for fe in fs:
                if fe.lower().endswith(('.png', '.jpg')):
                    file_name = fe.split(".")[0]
                    file_path = os.path.join(r, fe)
                    self.det_res = self.det.run(file_path)
                    res = self.det_res
                    save_path = os.path.join(save_base, file_name + ".txt")
                    with open(save_path, 'w') as write:
                        for dr in range(len(res)):
                            x1 = int(res[dr]['x1'])
                            x2 = int(res[dr]['x2'])
                            y1 = int(res[dr]['y1'])
                            y2 = int(res[dr]['y2'])
                            # 小框剔除
                            if x2 - x1 < 65 or y2 - y1 < 65:
                                continue
                            class_id = res[dr]['class_id']
                            conf = res[dr]['confidence']
                            x_center = round(float(res[dr]["x_center"]), 6)
                            y_center = round(float(res[dr]["y_center"]), 6)
                            w = round(float(res[dr]["w"]), 6)
                            h = round(float(res[dr]["h"]), 6)

                            w_str = "{} {} {} {} {} {}\n".format(int(class_id), x_center, y_center, w, h, round(conf, 6))
                            write.write(w_str)

    def test_det_bulk(self):
        # 批量检测 并 存小图
        img_base_path = r"E:\fish\comp\test\test\images"
        save_base = r"E:\fish\comp\origin\train\crop"
        for r, d, fs in os.walk(img_base_path):
            for fe in fs:
                if fe.lower().endswith(('.png', '.jpg')):
                    file_name = fe.split(".")[0]
                    file_path = os.path.join(r, fe)
                    img = cv2.imread(file_path)
                    self.det_res = self.det.run(file_path)
                    res = self.det_res
                    for dr in range(len(res)):
                        x1 = int(res[dr]['x1'])
                        x2 = int(res[dr]['x2'])
                        y1 = int(res[dr]['y1'])
                        y2 = int(res[dr]['y2'])
                        # 小框剔除
                        if x2 - x1 < 65 or y2 - y1 < 65:
                            continue
                        # 边缘剔除
                        if x1 == 0 or y1 == 0 or x2 == (x2 - x1) or y2 == (y2 - y1):
                            continue
                        class_id = res[dr]['class_id']
                        save_dir = os.path.join(save_base, str(int(class_id)))
                        if not os.path.exists(save_dir):
                            os.makedirs(save_dir)
                        crop = img[y1:y2, x1:x2]
                        save_path = os.path.join(save_dir, file_name + str(dr) + ".png")
                        print(save_path)
                        cv2.imwrite(save_path, crop)

    def test_catch(self):
        # 测试特征提取
        self.test_det()
        self.test_init_catch()
        one_img = self.det_res[0]
        img = cv2.imread(self.img_path)
        y1 = int(one_img["y1"])
        x1 = int(one_img["x1"])
        y2 = int(one_img["y2"])
        x2 = int(one_img["x2"])
        # print(x1, x2, y1, y2)
        crop = img[y1:y2, x1:x2]
        # cv2.imshow("1",crop)
        # cv2.waitKey(0)
        feature = self.cat.run(crop)
        print(feature.shape)

    def test_fai_study_one(self):
        # 测试学习一张
        self.test_init_fai()
        self.test_init_catch()
        img = cv2.imread(self.small_path)
        feature = self.cat.run(img)
        item = {}
        item['class_name'] = "19"
        self.fai.study([feature], [item])
        self.fai.to_disk(self.index_path, self.idmap_path)

    def test_fai_search_one(self):
        # 测试搜图
        self.test_init_fai()
        self.test_init_catch()
        img = cv2.imread(self.small_path_2)
        feature = self.cat.run(img)
        item = self.fai.run(feature)
        print(item)

    def test_fai_study_bul(self):
        # 批量学习
        self.test_init_fai()
        self.test_init_catch()
        base_path = r"F:\comptition\adjust_class\adjust_class"
        for root, dirs, files in os.walk(base_path):
            for i in dirs:
                if i == "待定":
                    continue
                class_name = i
                img_base_path = os.path.join(root, class_name)
                for r, d, fs in os.walk(img_base_path):
                    for fe in fs:
                        if fe.lower().endswith(('.png', '.jpg')):
                            file_path = os.path.join(r, fe)
                            img = cv2.imread(file_path)
                            print(file_path)
                            feature = self.cat.run(img)
                            item = {}
                            item['class_name'] = str(class_name)
                            self.fai.study([feature], [item])
                            self.fai.to_disk(self.index_path, self.idmap_path)
