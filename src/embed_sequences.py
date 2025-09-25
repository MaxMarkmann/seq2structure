import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from dataloader import load_cb513
from config import PROCESSED_DATA_DIR


def embed_protbert(seq: str, model, tokenizer, device="cpu"):
    """Embed a single protein sequence with ProtBERT."""
    spaced_seq = " ".join(list(seq))
    inputs = tokenizer(spaced_seq, return_tensors="pt", add_special_tokens=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.squeeze(0)
    return embeddings[1:-1].cpu().numpy()


def run_embedding(n_proteins=None, out_file="embeddings.npz", device="cpu"):
    print(f"Loading ProtBERT on {device} ...")
    tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
    model = AutoModel.from_pretrained("Rostlab/prot_bert").to(device)

    df = load_cb513(n_proteins=n_proteins)
    print("Dataset:", df.shape)

    all_embeddings = []
    all_labels = []
    all_ids = []

    for i, row in df.iterrows():
        seq_id = f"{row['pdb_id']}_{row['chain_code']}"
        seq = row["seq"]
        label = row["sst3"]  # oder "sst8" falls du das willst

        print(f"[{i+1}/{len(df)}] Embedding {seq_id} (len={len(seq)})")

        spaced_seq = " ".join(list(seq))
        tokens = tokenizer(spaced_seq, return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            embedding = model(**tokens).last_hidden_state.mean(dim=1).cpu().numpy()

        all_embeddings.append(embedding.squeeze())
        all_labels.append(label)
        all_ids.append(seq_id)

    all_embeddings = np.stack(all_embeddings, axis=0)

    # Pfad sauber ins processed-Verzeichnis schreiben
    out_path = PROCESSED_DATA_DIR / out_file
    np.savez_compressed(
        out_path,
        embeddings=all_embeddings,
        labels=np.array(all_labels, dtype=object),
        ids=np.array(all_ids, dtype=object),
    )
    print(f"Saved embeddings to {out_path}")
