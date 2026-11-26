# Pretrain a language model with Common Corpus

This example uses the public GPT-2 configuration and tokenizer, initializes random weights, and trains a causal language model with Gleeful Transformer. The short run demonstrates checkpointing and next-token continuation. A conversational assistant would need instruction fine-tuning after pretraining. [GPT-2](https://huggingface.co/openai-community/gpt2) · [Common Corpus](https://huggingface.co/datasets/PleIAs/common_corpus)

## Prepare and pack the data

Install Gleeful Transformer from the repository root with `python -m pip install -e .`, then run these commands from this example directory.

```sh
python data.py --train_cases 1024 --eval_cases 64 --shards 8
python tok.py --block_size 256 --shards 8
```

`data.py` streams documents, removes duplicate selected texts, and uses a seeded split to hold out evaluation documents. The dataset can require substantial bandwidth. To select source categories, pass `--datasets` with one or more of `OpenScience`, `OpenWeb`, `OpenCulture`, `OpenSource`, `OpenGovernment`, and `OpenSemantic`. The selected categories, dataset commit, and split seed are recorded in `source.json`. Pass `--dataset_revision` to repeat preparation against that commit.

You can also skip `data.py` and pack your own train and evaluation directories. Put one or more `.jsonl` files in each directory, with one `{"text": "Document text"}` object per line. Keep the evaluation documents separate from training. This command uses smaller blocks for a quick run on local text.

```sh
python tok.py --untok_train path/to/train --untok_eval path/to/eval \
  --block_size 32 --shards 1
```

`tok.py` inserts an end-of-document token between documents and packs each split into fixed-length blocks without padding. It drops each split's last incomplete block. The `gleeful_details.json` files record the packed paths, source model revision, block size, and available next-token targets. The tokenizer and model configuration are saved in `data/tok/train/model_source/`. To use another architecture, pass `--model_name` and, for a Hub model, `--model_revision` when packing. A local model configuration and tokenizer directory also works for offline runs.

If the tokenizer has no chat template, the example adds a content-only template to the saved tokenizer so `GleefulChat` can load and sample it. The training data is packed from the raw document text.

## Train on one device

With the 256-token blocks from the Common Corpus command, run the following on a GPU.

```sh
python train.py --run_name corpus-1gpu --bs 2 --gas 8 --token_budget 261120 \
  --checkpoint_steps 16 --eval_steps 16 --log_steps 8 --save_model_path model/1gpu
python sample.py --model_name model/1gpu --text "Artificial intelligence" --max_len 32
```

Each optimizer update uses 16 blocks. Every block contributes 255 next-token targets, giving 64 updates for this budget. The script rounds the budget down to a complete optimizer step. If the packed split is too small, prepare more text or lower the budget.

For a four-update smoke run with the 32-token local blocks, use the following command after packing at least four training blocks and one evaluation block. A tiny local model configuration also makes this run practical on CPU.

```sh
python train.py --run_name corpus-smoke --bs 1 --gas 1 --token_budget 124 \
  --checkpoint_steps 1 --eval_steps 1 --log_steps 1 --save_model_path model/smoke
python sample.py --model_name model/smoke --text "Artificial intelligence" --max_len 8
```

## Train with FSDP2

Pack the 256-token blocks before launching workers. Run each command on a single machine with enough BF16-capable CUDA GPUs visible to PyTorch and with the packed data accessible to all workers. `torchrun --nproc-per-node` sets the number of GPU workers. Both commands use the same `fsdp2.json`, which wraps `GPT2Block` layers because `tok.py` defaults to the GPT-2 tokenizer and configuration. `train.py` initializes the model with random weights from that configuration. [PyTorch torchrun](https://docs.pytorch.org/docs/main/distributed/elastic/run.html)

Each recipe processes 16 blocks per optimizer step (`GPU count × --bs × --gas`), or 4,080 next-token targets. The 261,120-target budget gives 64 steps. Add `--precision bf16` to the one-GPU command when comparing throughput.

### Two GPUs

The 64-step Common Corpus run and checkpoint resume were validated on two GPUs. A smaller GPT-2 run also verified FSDP2 training and resume on four GPUs.

```sh
torchrun --standalone --nproc-per-node=2 train.py --fsdp_config fsdp2.json --precision bf16 \
  --run_name corpus-fsdp2-2gpu --bs 2 --gas 4 --token_budget 261120 \
  --checkpoint_steps 16 --eval_steps 16 --log_steps 8 --save_model_path model/fsdp2-2gpu
```

### Eight GPUs

This launch example is unverified on eight GPUs.

```sh
torchrun --standalone --nproc-per-node=8 train.py --fsdp_config fsdp2.json --precision bf16 \
  --run_name corpus-fsdp2-8gpu --bs 2 --gas 1 --token_budget 261120 \
  --checkpoint_steps 16 --eval_steps 16 --log_steps 8 --save_model_path model/fsdp2-8gpu
```

For a different architecture, repack with `tok.py --model_name` and pass an FSDP2 config whose `transformer_layer_cls_to_wrap` matches that model. Use the sharded checkpoints to resume training and the consolidated export in `model/fsdp2-*` for sampling. [Hugging Face FSDP2 configuration](https://huggingface.co/docs/transformers/en/fsdp)

## Resume and inspect a run

To resume, repeat the launch command with the same packed data, training settings, and process count. Add `--resume_path runs/<run_name>/checkpoint-N`, where `N` is a saved step before the end of training. The resumed run writes to a new run directory. Reusing the run name adds a numeric suffix.

Open `runs/<run_name>/loss_vs_tokens.png` to compare training and held-out loss. The run's `logs.jsonl` records evaluation loss, cumulative non-padding input tokens, input tokens per second, and checkpoint paths. `train.py` prints the final held-out loss and saves the model and tokenizer to `--save_model_path`.

To upload a saved export to a private Hugging Face model repository, sign in and run `python push.py --save_model_path model/1gpu --repo_id your-account/your-model`.

## Troubleshoot deployment

| What goes wrong | What to do |
| --- | --- |
| PyTorch sees too few GPUs, or NCCL is unavailable. | Check `nvidia-smi` and `CUDA_VISIBLE_DEVICES`. Use a CUDA build of PyTorch with NCCL and set `--nproc-per-node` to the number of visible GPUs you want to use. |
| NCCL hangs or reports a communication error. | Verify that the PyTorch container and host NVIDIA driver are compatible. To check whether direct GPU peer-to-peer communication is the issue, retry with `NCCL_P2P_DISABLE=1` before `torchrun`. [NVIDIA NCCL troubleshooting](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting/gpu_troubleshooting.html) |
| Files disappear after a restart, or checkpoint resume fails. | Store packed data and runs on persistent storage at a consistent mount path. Resume a completed intermediate `checkpoint-N` with the original GPU count and precision. |
