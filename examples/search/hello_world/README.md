# Hello World with Gleeful Search

Build a vector index from generated documents and query it locally. From the repository root, install with `python -m pip install -e '.[search]'`. Then run the command from this directory.

```sh
python push.py --query "Find case 3"
```

The script writes sample documents to `data/raw/train.jsonl` and saves the index to `data/output/`. It uses 100 documents by default. For a quick run, pass `--cases 8`. Use a fresh `--out_dir` when rebuilding with a different document count. The first run may download the embedding and ranking models.

To publish the index to a private Hugging Face dataset, sign in and pass your dataset ID.

```sh
hf auth login
python push.py --repo_id your-account/gleeful-search-hello-world
```
