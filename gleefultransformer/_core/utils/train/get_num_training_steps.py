import math


def get_num_training_steps(
    train_cases: int,
    num_devices: int,
    epochs: float,
    gas: int,
    bs: int,
) -> int:
    batches_per_epoch = math.ceil(train_cases / (num_devices * bs))
    full_epochs, final_batches = divmod(math.ceil(batches_per_epoch * epochs), batches_per_epoch)
    return full_epochs * math.ceil(batches_per_epoch / gas) + math.ceil(final_batches / gas)
