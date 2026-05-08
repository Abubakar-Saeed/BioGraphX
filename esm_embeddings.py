import torch
import pandas as pd
import numpy as np
import os
import gc
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# ==========================================
# CONFIGURATION
# ==========================================
CSV_PATH = "/kaggle/input/embeddingdeeploc/Only_Main_Unmatched.csv"
OUTPUT_DIR = "esm_embeddings_compressed_2"
MODEL_NAME = "facebook/esm2_t36_3B_UR50D"

# ---------------------------------------------------
#⚙️ SPLITTING CONFIGURATION (Control this!)
# ---------------------------------------------------
# Set to 1 for first run, 2 for second run
PART_TO_RUN = 1  
TOTAL_PARTS = 1  

# ---------------------------------------------------
# 🚀 GPU CONFIGURATION
# ---------------------------------------------------
# Batch size 4 is safe for 2x T4s (2 proteins per GPU)
# If you get OOM, reduce to 2.
BATCH_SIZE = 15  
NUM_WORKERS = 2

class ProteinDataset(Dataset):
    def __init__(self, df):
        self.entry_ids = df['ACC'].values
        self.sequences = df['Sequence_main'].values

    def __len__(self):
        return len(self.entry_ids)

    def __getitem__(self, idx):
        return self.entry_ids[idx], self.sequences[idx]

def extract_esm_embeddings_optimized():
    # 1. Setup Output
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"📂 Saving to: {OUTPUT_DIR}")

    # 2. Load Data & Split
    df = pd.read_csv(CSV_PATH)
    total_seqs = len(df)
    
    # --- SPLITTING LOGIC ---
    # Split dataframe into equal chunks based on TOTAL_PARTS
    chunks = np.array_split(df, TOTAL_PARTS)
    df_part = chunks[PART_TO_RUN - 1] # Select the specific part
    
    print(f"📊 Total Dataset: {total_seqs}")
    print(f"🔄 Running PART {PART_TO_RUN}/{TOTAL_PARTS}")
    print(f"📉 Processing {len(df_part)} sequences in this run...")

    # 3. Model Setup (Multi-GPU)
    print(f"🚀 Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME, torch_dtype=torch.float16)
    
    # Enable DataParallel if multiple GPUs exist
    if torch.cuda.device_count() > 1:
        print(f"🔥 {torch.cuda.device_count()} GPUs detected! Activating DataParallel.")
        model = torch.nn.DataParallel(model)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    # 4. Dataset & Loader
    dataset = ProteinDataset(df_part)
    
    # Custom Collate to tokenize & pad batch
    def collate_fn(batch):
        ids, seqs = zip(*batch)
        # Tokenize batch (padding is crucial for batch processing)
        inputs = tokenizer(
            list(seqs), 
            return_tensors="pt", 
            truncation=True, 
            max_length=1024, 
            padding=True 
        )
        return ids, inputs

    loader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True
    )

    # 5. Extraction Loop
    print("⚡ Starting Optimized Extraction...")
    
    with torch.no_grad():
        for batch_ids, inputs in tqdm(loader, desc=f"Part {PART_TO_RUN}"):
            # Move inputs to GPU
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)
            
            # Forward Pass (Distributed across GPUs automatically)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Get embeddings (Batch, Seq_Len, Dim)
            # .module handles the DataParallel wrapper if present
            embeddings = outputs.last_hidden_state.detach().cpu()
            
            # Save individual files
            mask = inputs['attention_mask'].cpu()
            
            for i, entry_id in enumerate(batch_ids):
                save_path = os.path.join(OUTPUT_DIR, f"{entry_id}.npz")
                
                # Check exist
                if os.path.exists(save_path): continue
                
                # Remove Padding (Use mask to slice only real data)
                # mask[i] is 1 for real token, 0 for pad
                valid_len = mask[i].sum().item()
                
                # Slice: [0 : valid_len] -> Convert to FP16 Numpy
                emb_numpy = embeddings[i, :valid_len, :].numpy().astype(np.float16)
                
                # Save
                np.savez_compressed(save_path, embedding=emb_numpy)

            # Cleanup
            del input_ids, attention_mask, outputs, embeddings
            gc.collect()

    print(f"✅ Part {PART_TO_RUN} Complete!")

if __name__ == "__main__":
    extract_esm_embeddings_optimized()