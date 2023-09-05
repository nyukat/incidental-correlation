from datasets.base_dataset import BaseDataset


class VQAv2Dataset(BaseDataset):
    def __init__(self, *args, split="", is_cp=False, **kwargs):
        assert split in ["train", "val", "test"]
        self.split = split
        self.is_cp = is_cp
        cp_str = "cp" if is_cp else ""

        if split == "train":
            names = [f"vqa{cp_str}v2_train"]
        elif split == "val":
            names = [f"vqa{cp_str}v2_val"]
        elif split == "test":
            names = [f"vqa{cp_str}v2_test"]

        super().__init__(
            *args,
            **kwargs,
            names=names,
            text_column_name="questions",
            remove_duplicate=False,
        )


    def __getitem__(self, index):
        image_tensor = self.get_image(index)["image"]
        text = self.get_text(index)["text"]

        index, question_index = self.index_mapper[index]
        qid = self.table["question_id"][index][question_index].as_py()

        if not self.is_cp and self.split == "test":
            answers = list()
            labels = list()
            scores = list()
        else:
            answers = self.table["answers"][index][question_index].as_py()
            labels = self.table["answer_labels"][index][question_index].as_py()
            scores = self.table["answer_scores"][index][question_index].as_py()

        return {
            "image": image_tensor,
            "text": text,
            "vqa_answer": answers,
            "vqa_labels": labels,
            "vqa_scores": scores,
            "qid": qid,
        }
