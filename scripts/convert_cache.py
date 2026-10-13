"""Convert an old list-of-tokens cache pickle to the memory-mapped format (run once, one process).

Usage: python scripts/convert_cache.py data_cache/tokenized_data_250000_200000000_vocab24000_tokenizers.pkl
"""
import pickle
import sys
from pathlib import Path

import numpy as np

path = Path(sys.argv[1])
with path.open("rb") as f:
    cached = pickle.load(f)
if cached["tokens"] is None:
    sys.exit("already converted")
tokens = np.asarray(cached["tokens"], dtype=np.int32)
dtype = np.uint16 if tokens.max() < 2**16 else np.int32
np.save(path.with_suffix(".tokens.npy"), tokens.astype(dtype))
cached["tokens"] = None
with path.open("wb") as f:
    pickle.dump(cached, f)
print(f"converted {len(tokens):,} tokens -> {path.with_suffix('.tokens.npy')} ({dtype.__name__})")
