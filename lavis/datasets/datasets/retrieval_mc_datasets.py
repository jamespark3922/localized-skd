import torch

from lavis.datasets.datasets.base_dataset import BaseDataset

class RetrievalMCDataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def collater(self, samples):
        (
            image_list,
            input_list,
            image_id_list,
            instance_id_list,
            label_list,
        ) = ([], [], [], [], [])

        for sample in samples:
            image_list.append(sample["image"])
            input_list.append(sample["text_input"])
            image_id_list.append(sample["image_id"])
            label_list.append(sample["label"])
            instance_id_list.append(sample["instance_id"])

        to_return = {
            "image": torch.stack(image_list, dim=0),
            "text_input": input_list,
            "image_id": torch.LongTensor(image_id_list),
            "instance_id": instance_id_list,
        }

        if label_list[0] is not None:
            to_return.update({
                "label": torch.LongTensor(label_list)
            })

        return to_return