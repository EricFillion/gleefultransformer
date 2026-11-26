import os
from typing import List

from gleefultransformer import CHATTokArgs, GleefulChat


def t_tok_length(
    gleeful_model: GleefulChat,
    input_data_list: List[str],
    out_dir: str,
    args: CHATTokArgs,
    target_length: int,
):
    save_path = os.path.join(out_dir, "tok_len_path")

    for input_data in input_data_list:
        gleeful_model.tok(input_data, save_path, args=args)

        tok_data = gleeful_model.tok_dir_to_dataset(save_path)

        for input_ids, labels in zip(tok_data["input_ids"], tok_data["input_ids"]):
            print(len(input_ids))
            print(len(labels))
            print(target_length)
            assert len(input_ids) == len(labels)
            assert target_length == len(input_ids)
