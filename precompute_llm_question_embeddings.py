# precompute_llm_question_embeddings.py

import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

from utils.DataLoader import get_link_classification_data

# ============================
# Config
# ============================
DATASET_NAME = "assist17"
VAL_RATIO = 0.1
TEST_RATIO = 0.1

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(device)
model.eval()


def load_question_texts():
    """
    Build synthetic question texts from the dataset itself.

    We use the first feature column of node_raw_features as a concept ID
    (same assumption as in your ReAKT/CGC code) and turn each node into a
    short natural-language description.

    This gives us:
        question_texts[node_id] = string

    Only question nodes (dst_node_ids) will really matter for ReAKT,
    but we generate a sentence for every node for simplicity.
    """
    data_tuple = get_link_classification_data(
        dataset_name=DATASET_NAME,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
    )
    node_raw_features = data_tuple[0]  # shape [num_nodes, D]
    num_nodes = node_raw_features.shape[0]

    # First column used as concept ID in your current ReAKT code
    concept_ids = node_raw_features[:, 0].astype(int)

    question_texts = []
    for node_id in range(num_nodes):
        cid = concept_ids[node_id]
        # Short, consistent description. You can tweak wording if you like.
        text = (
            f"In the ASSISTments 2017 knowledge tracing dataset, "
            f"this item is a question node {node_id} associated with "
            f"knowledge component {cid}."
        )
        question_texts.append(text)

    return question_texts


def encode_batch(texts):
    with torch.no_grad():
        toks = tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=128,
        ).to(device)
        out = model(**toks)
        # Simple mean pooling over token embeddings
        emb = out.last_hidden_state.mean(dim=1)  # [B, hidden_dim]
        return emb.cpu().numpy()


def main():
    question_texts = load_question_texts()
    num_nodes = len(question_texts)
    print(f"Loaded synthetic texts for {num_nodes} nodes.")

    batch_size = 64
    all_embs = []

    for start in range(0, num_nodes, batch_size):
        end = min(start + batch_size, num_nodes)
        batch_texts = question_texts[start:end]
        batch_embs = encode_batch(batch_texts)
        all_embs.append(batch_embs)
        print(f"Encoded nodes {start}–{end - 1}")

    node_raw_features_llm = np.vstack(all_embs)  # [num_nodes, hidden_dim]
    print("Final LLM embedding shape:", node_raw_features_llm.shape)

    out_path = f"{DATASET_NAME}_reakt_llm_node_features.npy"
    np.save(out_path, node_raw_features_llm)
    print(f"Saved LLM features to {out_path}")


if __name__ == "__main__":
    main()
