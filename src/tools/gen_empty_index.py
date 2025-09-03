import faiss
import numpy as np


def build_faiss_index_v2_empty(d):
    # d = 1536  # 特征维度
    base_index = faiss.IndexFlatL2(d)  # 基础 L2 索引
    index = faiss.IndexIDMap(base_index)  # 包装 IDMap
    return index


def save_empty_index(index, index_path, idmap_path):
    """
    保存空索引和空 ID 映射文件
    """
    # 保存空的 faiss 索引
    faiss.write_index(index, index_path)

    # 保存空的 ID 映射文件
    with open(idmap_path, 'w', encoding='utf-8') as f:
        pass  # 空文件


if __name__ == "__main__":
    # 建一个flat2暴力搜索index

    index_path = r"F:\comptition\model\index.bin"
    idmap_path = r"F:\comptition\model\idmap.txt"
    index = build_faiss_index_v2_empty(1280)
    save_empty_index(index, index_path, idmap_path)
