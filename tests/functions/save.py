from gleefultransformer import CHATCallArgs, GleefulChat



def t_save(
    gleeful_model: GleefulChat, gleeful_type: type[GleefulChat], call_input: str,
    save_path: str, args: CHATCallArgs | None,
):
    gleeful_model.save(save_path)
    reloaded = gleeful_type(model_name=save_path)
    kwargs = {"args": args} if args is not None else {}
    assert gleeful_model(call_input, **kwargs).text == reloaded(call_input, **kwargs).text
