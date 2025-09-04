from ultralytics import YOLO


class DetYoloV8(object):
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def run(self, img_path):
        results = self.model(img_path)
        rsp_list = []
        for result in results:
            # 获取边界框信息
            boxes = result.boxes

            # 遍历每个检测到的目标
            for box in boxes:
                tmp = {}

                x1, y1, x2, y2 = box.xyxy[0].tolist()  # 原始图片尺寸坐标，用于截图
                x_center, y_center, w, h = box.xywhn[0].tolist()  # 归一化后的坐标，用于结果输出

                # 获取置信度
                confidence = box.conf[0].item()

                # 获取类别ID和类别名称
                class_id = box.cls[0].item()
                # class_name = model.names[int(class_id)]
                tmp["x1"] = x1
                tmp["y1"] = y1
                tmp["x2"] = x2
                tmp["y2"] = y2
                tmp["x_center"] = x_center
                tmp["y_center"] = y_center
                tmp["w"] = w
                tmp["h"] = h
                tmp["confidence"] = confidence
                tmp["class_id"] = class_id
                rsp_list.append(tmp)

        return rsp_list
