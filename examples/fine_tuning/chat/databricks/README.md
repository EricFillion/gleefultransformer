# Fine-tune a chat model for instruction following

This example uses [Databricks Dolly](https://huggingface.co/datasets/databricks/databricks-dolly-15k). The default model is SmolLM2-135M-Instruct. If you choose another model, use one with a chat template.

Follow the [fine-tuning guide](../../README.md) to train the full model. To include background context with a sample prompt, pass `--context`.

```sh
python sample.py --context "Gleeful Transformer can run and fine-tune open-weight chat models." --text "What could I use Gleeful Transformer for?"
```

## LoRA and QLoRA

From the repository root, run `python -m pip install '.[qlora]'` for either LoRA or QLoRA. QLoRA uses bitsandbytes to load the base model in 4-bit NF4 on a CUDA GPU. Start with an unquantized base model for either mode.

From this example directory, run either command.

```sh
python train.py --adapter lora --train_cases 128 --eval_cases 32 --max_seq_len 512 --save_model_path outputs/model-lora/
python train.py --adapter qlora --train_cases 128 --eval_cases 32 --max_seq_len 512 --save_model_path outputs/model-qlora/
```

Use `--max_seq_len` to set the number of tokens each formatted conversation is padded or truncated to. A smaller value reduces GPU memory use.

Both modes accept `--lora_rank`, `--lora_alpha`, `--lora_dropout`, and `--lora_target_modules`. The defaults are rank 16, alpha 32, dropout 0.05, and `all-linear`. To choose specific layers, pass their names as separate arguments, for example `--lora_target_modules q_proj v_proj`.

The QLoRA example uses GleefulChat's `quantization="nf4"` preset. It enables double quantization and uses native BF16 compute when available or FP16 otherwise. For custom settings, pass `--qlora_quantization_config` with the path to a JSON object containing native Transformers `BitsAndBytesConfig` fields. The example enables 4-bit loading, and other omitted fields use Transformers defaults. [Transformers bitsandbytes configuration](https://huggingface.co/docs/transformers/en/quantization/bitsandbytes)

```json
{
  "bnb_4bit_quant_type": "nf4",
  "bnb_4bit_use_double_quant": false,
  "bnb_4bit_compute_dtype": "float16"
}
```

To generate from a saved adapter, pass its directory as `--model_name`. The original base model must be available. Use `--checkpoint` to sample the latest checkpoint under `--out_dir`. When continuing training with `--resume_path`, the checkpoint supplies the model and adapter settings.

```sh
python sample.py --model_name outputs/model-lora/
python sample.py --model_name outputs/model-qlora/
```
