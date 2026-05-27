from langchain_core.embeddings import Embeddings
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForFeatureExtraction
from typing import List
from tqdm import tqdm

class StellaEmbeddings(Embeddings):
    """
    Stella-en-1.5B-v5 wrapper using Optimum ONNX Runtime for CPU optimization.
    """

    def __init__(self, model_name: str = "infgrad/stella_en_1.5B_v5", batch_size: int = 8):
        """
        Initialize Stella embeddings using ONNX with local caching.
        """
        import os
        self.batch_size = batch_size
        local_onnx_dir = "./stella_onnx_cache"
        
        # Load the tokenizer (this is always fast)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Check if we already compiled it to ONNX previously
        if not os.path.exists(local_onnx_dir):
            print(f"\n[ONNX] First run detected! Converting {model_name} to ONNX.")
            print("[ONNX] This will take a few minutes. Ignore the TracerWarnings...")
            
            # 1. Download and convert
            self.model = ORTModelForFeatureExtraction.from_pretrained(
                model_name, 
                export=True, 
                provider="CPUExecutionProvider" 
            )
            
            # 2. Save the compiled model and tokenizer locally so we never do this again
            print(f"[ONNX] Saving compiled model to {local_onnx_dir}...")
            self.model.save_pretrained(local_onnx_dir)
            self.tokenizer.save_pretrained(local_onnx_dir)
            print("[ONNX] Save complete! Starting embeddings...\n")
            
        else:
            # Load instantly from the local compiled folder
            self.model = ORTModelForFeatureExtraction.from_pretrained(
                local_onnx_dir, 
                export=False, # We set this to False now!
                provider="CPUExecutionProvider" 
            )
            
        self.query_prefix = (
            "Instruct: Given a user query, retrieve relevant documents.\nQuery: "
        )

    def _encode(self, texts: List[str]) -> List[List[float]]:
        """Encode texts using last-token pooling + L2 normalization (in batches)."""
        all_embeddings = []
        
        # Real-time progress bar with ETA smoothing
        for i in tqdm(
            range(0, len(texts), self.batch_size), 
            desc="Embedding chunks (ONNX)", 
            unit="batch",
            smoothing=0.1
        ):
            batch_texts = texts[i : i + self.batch_size]
            
            # 1. Tokenize as normal
            encoded = self.tokenizer(
                batch_texts, 
                padding=True, 
                truncation=True, 
                max_length=8192,
                return_tensors="pt"
            )

            # --- ADD THESE 3 LINES ---
            # ONNX strictly requires position_ids. We generate them by 
            # keeping a running total of the attention mask and zeroing out padding.
            position_ids = encoded["attention_mask"].cumsum(dim=1) - 1
            position_ids.masked_fill_(encoded["attention_mask"] == 0, 0)
            encoded["position_ids"] = position_ids
            # -------------------------

            # 2. Pass the updated dictionary to the model
            outputs = self.model(**encoded)
            
            # Last-token pooling
            attention_mask = encoded["attention_mask"]
            last_hidden = outputs.last_hidden_state
            
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden.shape[0]
            pooled = last_hidden[torch.arange(batch_size), sequence_lengths]
            
            # L2 normalize
            normalized = F.normalize(pooled, p=2, dim=1)
            
            # Convert directly to list
            all_embeddings.extend(normalized.tolist())

        return all_embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents RAW - NO PREFIX."""
        return self._encode(texts)

    def embed_query(self, text: str) -> List[float]:
        """Embed query WITH instruction prefix."""
        prefixed_text = self.query_prefix + text
        return self._encode([prefixed_text])[0]