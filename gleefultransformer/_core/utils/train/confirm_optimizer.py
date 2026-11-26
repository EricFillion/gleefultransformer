from torch.optim import SGD, AdamW

from gleefultransformer._core.args import GleefulTrainArgs


def get_optim(args: GleefulTrainArgs, model, logger):
    parameters = model.parameters()
    if getattr(model, "peft_config", None):
        parameters = (parameter for parameter in parameters if parameter.requires_grad)

    if args.optim == "adamw":
        logger.debug("Using PyTorch's AdamW optimizer.")
        return AdamW(parameters, lr=args.lr, weight_decay=0.05)

    elif args.optim == "sgd":
        logger.debug("Using PyTorch's SGD optimizer")
        return SGD(
            parameters, lr=args.lr, momentum=0, weight_decay=1e-4
        )

    else:
        raise ValueError(f"Unsupported optimizer: {args.optim}")
