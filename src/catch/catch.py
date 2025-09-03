import timm
import torch
import numpy as np
from PIL import Image

class Catch(object):
    def __init__(self):
        self.device = torch.device("cpu")
        model_name = 'mobilenetv3_large_100'  # 示例：Large 变体
        self.model = timm.create_model(model_name, pretrained=True, num_classes=0,features_only=False)  # 自动下载权重
        self.model.eval().to(self.device)
        self.config = timm.data.resolve_data_config({}, model=self.model)
        self.transform = timm.data.create_transform(**self.config)
        self.feat_dim = getattr(self.model, 'num_features', None)
        # 输出维度（num_features），用于 FAISS 维度设定
        if self.feat_dim is None:
            # 回退：做一次前传探测维度
            x = torch.zeros(1, 3, self.config.get('input_size', (3, 192, 192))[1],
                            self.config.get('input_size', (3, 192, 192))[2])
            with torch.no_grad():
                y = self.model(x)
            self.feat_dim = y.shape[-1]

    @torch.inference_mode()
    def run(self, images: np.ndarray, batch_size: int = 16) -> np.ndarray:
        tensors = []
        for img in images:
            if isinstance(img, np.ndarray):
                # 保证 RGB
                if img.ndim == 2:
                    img = np.stack([img] * 3, axis=-1)
                if img.shape[2] == 3:
                    # 假设是 BGR（常见 OpenCV），转 RGB
                    img = img[:, :, ::-1]
                img = Image.fromarray(img)
            tensors.append(self.transform(img))
        dataset = torch.stack(tensors, dim=0)

        feats = []
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i + batch_size].to(self.device)
            vec = self.model(batch)  # (B, D)
            feats.append(vec.cpu())
        feats = torch.cat(feats, dim=0)[0]
        feature = feats.cpu().detach().numpy().astype(np.float32)
        feature = np.squeeze(feature)
        return feature
