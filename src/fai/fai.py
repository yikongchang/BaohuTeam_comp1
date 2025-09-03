import faiss
import numpy as np


class IDMapItem:
    id_: int
    class_name: str
    dt: str


class Index(object):
    def __init__(self, faiss_data, idmap_data):
        self.idmap = {}
        self.index = faiss.deserialize_index(np.frombuffer(faiss_data, dtype=np.uint8))
        self.idmap_tmp = idmap_data.decode('utf-8').split("\n")
        for line in self.idmap_tmp:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",", 2)
            item = {}
            item['class_name'] = parts[1].strip()
            item["id"] = int(parts[0])
            self.idmap[item["id"]] = item

    def run(self, feature):
        output_np = feature.reshape(1, -1).astype('float32')
        distances, ids = self.index.search(output_np, k=1)  # flat2索引，先只找一个
        distances = distances[0, :]
        ids = ids[0, :]
        items = []
        for i in range(len(ids)):
            distance = distances[i]
            id_ = ids[i]
            im_item = self.idmap[id_]
            item = {}
            item['distance'] = distance.item()
            item['id'] = id_.item()
            item['class_name'] = im_item["class_name"]
            items.append(item)
        return items

    def study(self, feature_list, idmap_list):

        ids, max_id, new_ids = self.generate_ids(len(feature_list))
        for idx, id_ in enumerate(ids):
            idmap_list[idx]["id"] = id_
        ids = np.asarray([item["id"] for item in idmap_list], dtype=np.int64)
        features = []
        for f in feature_list:
            features.append(f)
        features = np.asarray(features, dtype=np.float32)
        self.index.add_with_ids(features, ids)
        for item in idmap_list:
            self.idmap[item["id"]] = item

    def to_disk(self, index_path, idmap_path):
        idmap_str = get_idmap_str(self.idmap)
        buf = faiss.serialize_index(self.index)
        buf = buf.tobytes()
        with open(index_path, "wb") as w:
            w.write(buf)
        with open(idmap_path, "w") as w:
            w.write(idmap_str)

    def generate_ids(self, n):
        ids = []
        max_id = -1
        new_ids = []
        for id_ in self.idmap:
            if id_ > max_id:
                max_id = id_
        for _ in range(n):
            id_ = max_id + 1
            new_ids.append(id_)
            max_id = id_
            ids.append(id_)
        return ids, max_id, new_ids


def get_idmap_str(idmap):
    lines = []
    for item in idmap.values():
        line = "%d,%s" % (item['id'], item['class_name'])
        lines.append(line)
    return "\n".join(lines)
