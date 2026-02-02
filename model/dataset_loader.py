import torch
from torch.utils.data import Dataset

NUC_TO_ID = {
    "A": 0,
    "C": 1,
    "G": 2,
    "U": 3,
    "PAD": 4,
}

PAD_ID = 4

class RNADataset(Dataset):
    def __init__(self, seq_df, label_df):
        self.seq_df = seq_df.reset_index(drop=True)
        self.label_df = label_df
        self.vocab = {"A": 0, "C": 1, "G": 2, "U": 3}

    def __len__(self):
        return len(self.seq_df)

    def __getitem__(self, idx):
        # ---- sequence ----
        row = self.seq_df.iloc[idx]
        target_id = row.target_id
        seq = row.sequence

        seq_encoded = torch.tensor(
            [self.vocab[c] for c in seq],
            dtype=torch.long
        )

        # ---- coordinates ----
        coords_df = self.label_df[
            self.label_df.split_ID == target_id
        ].sort_values("resid")

        coords = torch.tensor(coords_df[["x_1","y_1","z_1"]].values, dtype=torch.float32)
        coords[~torch.isfinite(coords)] = 0.0

        # ---- sanity check ----
        if len(seq_encoded) != coords.shape[0]:
            raise ValueError(
                f"Length mismatch for {target_id}: "
                f"{len(seq_encoded)} vs {coords.shape[0]}"
            )

        return seq_encoded, coords
    
class RNATestDataset(Dataset):
    def __init__(self, test_sequences_df):
        self.df = test_sequences_df.reset_index(drop=True)
        self.vocab = {"A": 0, "C": 1, "G": 2, "U": 3}

    def __len__(self):
        return len(self.df)

    def encode_sequence(self, seq):
        return torch.tensor([self.vocab[c] for c in seq], dtype=torch.long)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        seq = row["sequence"]
        tokens = self.encode_sequence(seq)
        length = len(tokens)

        mask = torch.ones(length, dtype=torch.bool)

        return {
            "tokens": tokens,
            "mask": mask,
            "target_id": row["target_id"],
            "sequence": seq      
        }
    
def rna_collate_fn(batch):
    sequences, coords = zip(*batch)

    lengths = torch.tensor([len(seq) for seq in sequences])

    B = len(sequences)
    L_max = max(lengths)

    tokens = torch.zeros(B, L_max, dtype=torch.long)
    coords_pad = torch.zeros(B, L_max, 3)

    for i, (seq, xyz) in enumerate(zip(sequences, coords)):
        L = len(seq)
        tokens[i, :L] = seq
        coords_pad[i, :L] = xyz

    mask = torch.arange(L_max)[None, :] < lengths[:, None]

    return {
        "tokens": tokens,
        "coords": coords_pad,
        "mask": mask,
        "lengths": lengths
    }

def rna_test_collate_fn(batch):
    B = len(batch)
    lengths = [len(item["tokens"]) for item in batch]
    max_len = max(lengths)

    tokens = torch.full((B, max_len), PAD_ID, dtype=torch.long)
    mask = torch.zeros((B, max_len), dtype=torch.bool)

    meta = []

    for i, item in enumerate(batch):
        L = lengths[i]
        tokens[i, :L] = item["tokens"]
        mask[i, :L] = item["mask"]

        meta.append({
            "target_id": item["target_id"],
            "length": L,
        })

    return {
        "tokens": tokens,
        "mask": mask,
        "meta": meta,
    }



