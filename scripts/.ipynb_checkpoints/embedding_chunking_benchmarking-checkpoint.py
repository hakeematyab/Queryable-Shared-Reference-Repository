
import os

cache_dir ='/scratch/hakeem.at/Queryable-Shared-Reference-Repository/notebooks/pretrained_models'

os.environ['HF_HOME'] = cache_dir 
os.environ['TRANSFORMERS_CACHE'] = cache_dir
os.environ['HUGGINGFACE_HUB_CACHE'] = cache_dir


import os
import time
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
import json
import random
import re
from tqdm.auto import tqdm

from langchain_community.document_loaders import PyPDFLoader, BSHTMLLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain.embeddings.base import Embeddings
from langchain_community.vectorstores import FAISS

import torch
from transformers import AutoTokenizer, AutoModel
from adapters import AutoAdapterModel
from multiprocessing import Pool, cpu_count


import torch
import gc

def clear_gpu(*items):
    for item in items:
        del item
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.synchronize()


seed = 42
random.seed(seed)

class Specter2Embeddings(Embeddings):
    def __init__(self, model_id = 'allenai/specter2_base', adapter="allenai/specter2", device="cuda", batch_size=1024):
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoAdapterModel.from_pretrained(model_id)
        self.model.load_adapter(adapter, source="hf", load_as="specter2", set_active=True)
        if device=="cuda":
            self.model = self.model.cuda()

    def embed_documents(self, texts):
        outputs = []
        self.model.eval()
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:min(i + self.batch_size, len(texts))]
            with torch.no_grad():
                inputs = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=512).to(self.model.device)
    
                output = self.model(**inputs)
                output = output.last_hidden_state[:,0,:].cpu().tolist()
                outputs.extend(output)
        return outputs

    def embed_query(self, text):
        inputs = self.tokenizer([text], padding=True, truncation=True, return_tensors="pt", max_length=512).to(self.model.device)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)
            outputs = outputs.last_hidden_state[:,0,:].cpu().tolist()[0]
        return outputs


class SciNCLEmbeddings(Embeddings):
    def __init__(self, model_id='malteos/scincl', device="cuda", batch_size=32):
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id)
        if device == "cuda":
            self.model = self.model.cuda()
    
    def embed_documents(self, texts):
        outputs = []
        self.model.eval()
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:min(i + self.batch_size, len(texts))]
            with torch.no_grad():
                inputs = self.tokenizer(batch, padding=True, truncation=True, 
                                       return_tensors="pt", max_length=512).to(self.model.device)
                output = self.model(**inputs)
                embeddings = output.last_hidden_state[:, 0, :].cpu().tolist()
                outputs.extend(embeddings)
        return outputs
    
    def embed_query(self, text):
        self.model.eval()
        inputs = self.tokenizer([text], padding=True, truncation=True, 
                               return_tensors="pt", max_length=512).to(self.model.device)
        with torch.no_grad():
            output = self.model(**inputs)
            embedding = output.last_hidden_state[:, 0, :].cpu().tolist()[0]
        return embedding

class SPECTEREmbeddings(Embeddings):
    def __init__(self, model_id='allenai/specter', device="cuda", batch_size=32):
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id)
        if device == "cuda":
            self.model = self.model.cuda()
    
    def embed_documents(self, texts):
        outputs = []
        self.model.eval()
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:min(i + self.batch_size, len(texts))]
            with torch.no_grad():
                inputs = self.tokenizer(batch, padding=True, truncation=True, 
                                       return_tensors="pt", max_length=512).to(self.model.device)
                output = self.model(**inputs)
                # CLS token pooling (HuggingFace version)
                embeddings = output.last_hidden_state[:, 0, :].cpu().tolist()
                outputs.extend(embeddings)
        return outputs
    
    def embed_query(self, text):
        self.model.eval()
        inputs = self.tokenizer([text], padding=True, truncation=True, 
                               return_tensors="pt", max_length=512).to(self.model.device)
        with torch.no_grad():
            output = self.model(**inputs)
            embedding = output.last_hidden_state[:, 0, :].cpu().tolist()[0]
        return embedding

class SciBERTEmbeddings(Embeddings):
    def __init__(self, model_id='allenai/scibert_scivocab_uncased', device="cuda", batch_size=32):
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id)
        if device == "cuda":
            self.model = self.model.cuda()
    
    def embed_documents(self, texts):
        outputs = []
        self.model.eval()
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:min(i + self.batch_size, len(texts))]
            with torch.no_grad():
                inputs = self.tokenizer(batch, padding=True, truncation=True, 
                                       return_tensors="pt", max_length=512).to(self.model.device)
                output = self.model(**inputs)
                embeddings = output.last_hidden_state[:, 0, :].cpu().tolist()
                outputs.extend(embeddings)
        return outputs
    
    def embed_query(self, text):
        self.model.eval()
        inputs = self.tokenizer([text], padding=True, truncation=True, 
                               return_tensors="pt", max_length=512).to(self.model.device)
        with torch.no_grad():
            output = self.model(**inputs)
            embedding = output.last_hidden_state[:, 0, :].cpu().tolist()[0]
        return embedding

class PubMedBERTEmbeddings(Embeddings):
    def __init__(self, model_id='microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext', 
                 device="cuda", batch_size=32):
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id)
        if device == "cuda":
            self.model = self.model.cuda()
    
    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def embed_documents(self, texts):
        outputs = []
        self.model.eval()
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:min(i + self.batch_size, len(texts))]
            with torch.no_grad():
                inputs = self.tokenizer(batch, padding=True, truncation=True, 
                                       return_tensors="pt", max_length=512).to(self.model.device)
                output = self.model(**inputs)
                embeddings = self._mean_pooling(output, inputs['attention_mask']).cpu().tolist()
                outputs.extend(embeddings)
        return outputs
    
    def embed_query(self, text):
        self.model.eval()
        inputs = self.tokenizer([text], padding=True, truncation=True, 
                               return_tensors="pt", max_length=512).to(self.model.device)
        with torch.no_grad():
            output = self.model(**inputs)
            embedding = self._mean_pooling(output, inputs['attention_mask']).cpu().tolist()[0]
        return embedding

embedding_classes = {
    'specter2': Specter2Embeddings,
    'scincl': SciNCLEmbeddings,
    'specter': SPECTEREmbeddings,
    'scibert': SciBERTEmbeddings,
    'pubmedbert': PubMedBERTEmbeddings
}



eval_qa_dataset_path = "eval_qa_data.json"
eval_text_dataset_path = "eval_text_data.json"

with open(eval_qa_dataset_path, "r") as f:
    eval_qa_dataset = json.load(f)

with open(eval_text_dataset_path, "r") as f:
    eval_text_dataset = json.load(f)


filtered_eval_qa_dataset = []
count = 0
for qa in eval_qa_dataset:
    if set(qa["question"].lower().strip().split()) and set(qa["excerpt"].lower().strip().split()):
        filtered_eval_qa_dataset.append(qa)
    else:
        count+=1
print(f"Found {count} empty datapoints")


ground_truth_docs = []

for qa in filtered_eval_qa_dataset:
    excerpt = set(qa["excerpt"].lower().strip().split())
    if excerpt:
        ground_truth_docs.append(excerpt)
    else:
        print("found empty: ",qa["excerpt"])


queries = [qa["question"] for qa in filtered_eval_qa_dataset]

def retrieval_metrics(retrieved_docs_list, ground_truth_docs, hit_rate_threshold=0.5, mrr_threshold=0.3):
    ious = []
    precisions = []
    recalls = []
    hits = []
    mrrs = []
    flattened_retrieved_docs = []
    
    for retrieved_docs in retrieved_docs_list:
        flattened_retrieved_doc = set()
        for retrieved_doc in retrieved_docs:
            doc = set(retrieved_doc.lower().strip().split())
            flattened_retrieved_doc.update(doc)
        flattened_retrieved_docs.append(flattened_retrieved_doc)

    for retrieved_docs, retrieved_doc, ground_truth_doc in zip(retrieved_docs_list, flattened_retrieved_docs, ground_truth_docs):
        intersection = len(retrieved_doc&ground_truth_doc)
        union = len(retrieved_doc | ground_truth_doc)
        iou = intersection/union if union>0 else 0
        precision = intersection/len(retrieved_doc) if len(retrieved_doc)>0 else 0
        recall = intersection/len(ground_truth_doc) if len(ground_truth_doc)>0 else 0

        hit = 0
        if recall >= hit_rate_threshold:
            hit=1
        mrr = 0
        for rank, doc in enumerate(retrieved_docs,start=1):
            doc = set(doc.lower().strip().split())
            overlap = len(doc&ground_truth_doc)/len(ground_truth_doc) if len(ground_truth_doc)>0 else 0
            if overlap>=mrr_threshold:
                mrr = 1.0/rank
                break
        
        ious.append(iou)
        precisions.append(precision)
        recalls.append(recall)
        hits.append(hit)
        mrrs.append(mrr)
    return {
        'recall_mean': np.mean(recalls),
        'recall_std': np.std(recalls),
        'precision_mean': np.mean(precisions),
        'precision_std': np.std(precisions),
        'iou_mean': np.mean(ious),
        'iou_std': np.std(ious),
        'hit_rate': np.mean(hits),
        'mrr_mean': np.mean(mrrs),
        'mrr_std': np.std(mrrs)
    }

# Iterate and initialize each model
for embedding_model_name, EmbeddingClass in embedding_classes.items():
    embedding_model_start_time = time.time()
    embedding_model = EmbeddingClass()
    
    query_embeddings = embedding_model.embed_documents(queries)


    chunking_strat_config = {
        "recursive_cs256_co0":RecursiveCharacterTextSplitter(
                    chunk_size = 256,
                    chunk_overlap = 0,
                    separators = ["\n\n", "\n", ". ", "? ", "! ", " ", ""]
        ),
        "recursive_cs500_co0":RecursiveCharacterTextSplitter(
                    chunk_size = 512,
                    chunk_overlap = 0,
                    separators = ["\n\n", "\n", ". ", "? ", "! ", " ", ""]
        ),
        "recursive_cs256_co25":RecursiveCharacterTextSplitter(
                    chunk_size = 256,
                    chunk_overlap = 25,
                    separators = ["\n\n", "\n", ". ", "? ", "! ", " ", ""]
        ),
        "recursive_cs500_co50":RecursiveCharacterTextSplitter(
                    chunk_size = 512,
                    chunk_overlap = 50,
                    separators = ["\n\n", "\n", ". ", "? ", "! ", " ", ""]
        ),
        "semantic": SemanticChunker(
                    embeddings = embedding_model,
                    buffer_size = 1,
                    breakpoint_threshold_type="percentile",
                    breakpoint_threshold_amount=90
                    
        )
    }

    eval_metrics = {}

    for strategy in chunking_strat_config:
        print(f"\n=== Strategy: {strategy} ===", flush=True)
        
        splitter = chunking_strat_config[strategy]
        chunked_datas = []
        metadatas = []
        
        start = time.time()
        for doc_idx, doc in enumerate(eval_text_dataset):
            chunks = splitter.split_text(doc["text"])
            chunked_datas.extend(chunks)
            metadatas.extend([{"chunk_id": f"{doc_idx}_{strategy}_{chunk_idx}", "source":doc["source"]} for chunk_idx in range(len(chunks))])
        print(f"⏱️ Chunking: {time.time() - start:.2f}s | Chunks: {len(chunked_datas)}", flush=True)
        start = time.time()
        vector_store = FAISS.from_texts(chunked_datas, embedding_model, metadatas=metadatas)
        print(f"⏱️ Embedding + FAISS indexing: {time.time() - start:.2f}s", flush=True)
        
        start = time.time()
        retrieved_docs = []
        for query_embedding in query_embeddings:
            retrieved_doc = vector_store.similarity_search_by_vector(query_embedding, k=5)
            retrieved_doc = [doc.page_content for doc in retrieved_doc]
            retrieved_docs.append(retrieved_doc)
        print(f"⏱️ Search (3 queries): {time.time() - start:.2f}s", flush=True)
        
        start = time.time()
        computed_metrics = retrieval_metrics(retrieved_docs, ground_truth_docs)
        print(f"⏱️ Metrics computation: {time.time() - start:.2f}s", flush=True)
        
        eval_metrics[strategy] = computed_metrics

    print(f"⏱️Emebdding Model({embedding_model_name}) {time.time() - embedding_model_start_time:.2f}s", flush=True)

    results_df = pd.DataFrame.from_dict(eval_metrics, orient="index")
    results_df.index.name = "strategy"
    results_df = results_df.reset_index()
    results_df

    results_df.to_csv(f"{embedding_model_name}_chunking_strategies_comparison.csv")