# %%
import os

cache_dir ='/scratch/hakeem.at/Queryable-Shared-Reference-Repository/notebooks/pretrained_models'

os.environ['HF_HOME'] = cache_dir 
os.environ['TRANSFORMERS_CACHE'] = cache_dir
os.environ['HUGGINGFACE_HUB_CACHE'] = cache_dir

import warnings
import logging
import os
import sys
from contextlib import contextmanager

warnings.filterwarnings("ignore")
os.environ['PYTHONWARNINGS'] = 'ignore'

logging.getLogger("docling").setLevel(logging.ERROR) 
logging.getLogger("docling.backend").setLevel(logging.ERROR)
logging.getLogger("docling.datamodel").setLevel(logging.ERROR)
logging.getLogger("docling_parse").setLevel(logging.ERROR)
logging.getLogger("PIL").setLevel(logging.ERROR)
logging.getLogger("pdfplumber").setLevel(logging.ERROR)
logging.getLogger("pdfminer").setLevel(logging.ERROR)

for logger_name in logging.Logger.manager.loggerDict.keys():
    if 'docling' in logger_name.lower() or 'pdf' in logger_name.lower():
        logging.getLogger(logger_name).setLevel(logging.CRITICAL)

@contextmanager
def suppress_stdout_stderr():
    null_file = open(os.devnull, 'w')
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = null_file
    sys.stderr = null_file
    try:
        yield
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        null_file.close()


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
from pydantic import BaseModel, Field

from langchain_community.document_loaders import PyPDFLoader, BSHTMLLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, TokenTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain.embeddings.base import Embeddings
from langchain_community.vectorstores import FAISS

import torch
import torch.nn.functional as F
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModel
from sentence_transformers import SentenceTransformer
# from adapters import AutoAdapterModel
from multiprocessing import Pool, cpu_count

# %%
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.backend.docling_parse_v2_backend  import DoclingParseV2DocumentBackend
from docling.chunking import HybridChunker
from docling_core.transforms.chunker.hierarchical_chunker import HierarchicalChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
import multiprocessing as mp

pipeline_options = PdfPipelineOptions()
pipeline_options.do_ocr = False
# %%
import torch
import gc

def clear_gpu(*items):
    for item in items:
        del item
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.synchronize()

# %%
seed = 42
random.seed(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)

# if torch.cuda.is_available():
#     torch.cuda.manual_seed(seed)
#     torch.backends.cudnn.deterministic = True

# %%
raw_data_path = '../raw_data/'
raw_data_dir_names = ['ethan', 'sai']
raw_data_dirs = [raw_data_path+raw_data_dir_name for raw_data_dir_name in raw_data_dir_names]

# %%
counter = Counter()
paths = {}
for dir in raw_data_dirs:
    for file in Path(dir).rglob("*"):
        if file.is_file():
            doc_format = file.suffix.lower() if file.suffix else "no_extension"
            counter[doc_format]+=1
            list_of_paths = paths.get(doc_format,[])
            list_of_paths.append(str(file))
            paths[doc_format] = list_of_paths

# %%
documents_summary_df = pd.DataFrame(counter.items(), columns=["Document Format", "Document Count"])
documents_summary_df

# %%
valid_extensions = ['.html','.pdf','.docx']
metadata_extension = ['.ris']
total_documents = documents_summary_df[documents_summary_df["Document Format"].isin(valid_extensions)]["Document Count"].sum()

print(f"Total Document Count: {total_documents}")

# %%
html_path = paths['.html'][0]
pdf_path = paths['.pdf'][0]
docx_path = paths['.docx'][0]

# %%
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
        """Mean pooling - recommended for PubMedBERT"""
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


# %%
class MPNetEmbeddings(Embeddings):
    def __init__(self, model_id='sentence-transformers/all-mpnet-base-v2', device="cuda", batch_size=32, normalize=True):
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id)
        self.normalize = normalize
        if device == "cuda":
            self.model = self.model.cuda()
    
    def _mean_pooling(self, model_output, attention_mask):
        """Mean pooling - take attention mask into account for correct averaging"""
        token_embeddings = model_output[0]  # First element contains all token embeddings
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
                embeddings = self._mean_pooling(output, inputs['attention_mask'])
                if self.normalize:
                    embeddings = F.normalize(embeddings, p=2, dim=1)
                outputs.extend(embeddings.cpu().tolist())
        return outputs
    
    def embed_query(self, text):
        self.model.eval()
        inputs = self.tokenizer([text], padding=True, truncation=True, 
                               return_tensors="pt", max_length=512).to(self.model.device)
        with torch.no_grad():
            output = self.model(**inputs)
            embedding = self._mean_pooling(output, inputs['attention_mask'])
            if self.normalize:
                embedding = F.normalize(embedding, p=2, dim=1)
        return embedding.cpu().tolist()[0]


class MiniLMEmbeddings(Embeddings):
    def __init__(self, model_id='sentence-transformers/all-MiniLM-L6-v2', device="cuda", batch_size=32, normalize=True):
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id)
        self.normalize = normalize
        if device == "cuda":
            self.model = self.model.cuda()
    
    def _mean_pooling(self, model_output, attention_mask):
        """Mean pooling - take attention mask into account for correct averaging"""
        token_embeddings = model_output[0]  # First element contains all token embeddings
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
                embeddings = self._mean_pooling(output, inputs['attention_mask'])
                if self.normalize:
                    embeddings = F.normalize(embeddings, p=2, dim=1)
                outputs.extend(embeddings.cpu().tolist())
        return outputs
    
    def embed_query(self, text):
        self.model.eval()
        inputs = self.tokenizer([text], padding=True, truncation=True, 
                               return_tensors="pt", max_length=512).to(self.model.device)
        with torch.no_grad():
            output = self.model(**inputs)
            embedding = self._mean_pooling(output, inputs['attention_mask'])
            if self.normalize:
                embedding = F.normalize(embedding, p=2, dim=1)
        return embedding.cpu().tolist()[0]

class EmbeddingGemmaEmbeddings(Embeddings):
    def __init__(self, model_id='google/embeddinggemma-300m', device="cuda", batch_size=32):
        self.model = SentenceTransformer(model_id, device=device)
        self.model.max_seq_length = 512
        self.batch_size = batch_size
        self.tokenizer = self.model.tokenizer
    
    def embed_documents(self, texts, prompt_name="Retrieval-document"):
        embeddings = self.model.encode(
            texts,
            prompt_name=prompt_name,
            batch_size=self.batch_size,
            convert_to_tensor=True
        )
        return embeddings.cpu().numpy().tolist()
    
    def embed_query(self, text):
        embedding = self.model.encode(
            [text],
            prompt_name="Retrieval-query",
            convert_to_tensor=True
        )
        return embedding.cpu().numpy().tolist()[0]

# %%

# %%
import json
eval_qa_dataset_path = "eval_qa_data.json"

with open(eval_qa_dataset_path, "r") as f:
    eval_qa_dataset = json.load(f)

# %%
filtered_eval_qa_dataset = []
count = 0
for qa in eval_qa_dataset:
    if set(qa["question"].lower().strip().split()) and set(qa["excerpt"].lower().strip().split()):
        filtered_eval_qa_dataset.append(qa)
    else:
        count+=1
print(f"Found {count} empty datapoints")

# %%
excerpt_dist = []
for qa in eval_qa_dataset:
    excerpt_dist.append(len(qa["excerpt"].split()))

# %%
ground_truth_docs = []

for qa in filtered_eval_qa_dataset:
    excerpt = set(qa["excerpt"].lower().strip().split())
    if excerpt:
        ground_truth_docs.append(excerpt)
    else:
        print("found empty: ",qa["excerpt"])

# %%
queries = [qa["question"] for qa in filtered_eval_qa_dataset]

# %%
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

embedding_classes = {
    'specter2': Specter2Embeddings,
    'scincl': SciNCLEmbeddings,
    'specter': SPECTEREmbeddings,
    'scibert': SciBERTEmbeddings,
    'pubmedbert': PubMedBERTEmbeddings,
    'mpnet':MPNetEmbeddings,
    'minilm':MiniLMEmbeddings,
    'gemma':EmbeddingGemmaEmbeddings,
    'gemma2048':EmbeddingGemmaEmbeddings,
    'minilml12':MiniLMEmbeddingsL12
}

converter = DocumentConverter(format_options={
    InputFormat.PDF: PdfFormatOption(
        pipeline_options = pipeline_options,
        backend = DoclingParseV2DocumentBackend
    )
})
loaded_docs = []
with suppress_stdout_stderr():
    for path in tqdm(paths['.pdf'], total=len(paths['.pdf'])):
        try:
            doc = converter.convert(source=path).document
            loaded_docs.append(doc)
        except Exception as e:
            # print(f"Error processing {path}: {e}")
            continue
# %%
chunking_strat_config = {
    'hierarchical':(HierarchicalChunker, {'merge_list_items':True}),
    'hybrid':(HybridChunker,{'merge_peers':True})
}

# %%
runtime_metrics = []
for embedding_model_name, EmbeddingClass in embedding_classes.items():
    max_tokens = 512
    embedding_model_start_time = time.time()
    embedding_model = EmbeddingClass()
    if embedding_model_name == "gemma2048":
        embedding_model.model.max_seq_length = 2048
        max_tokens = 2048
    tokenizer = HuggingFaceTokenizer(
        tokenizer = embedding_model.tokenizer,
        max_tokens = max_tokens
    )
    if "gemma" in embedding_model_name:
        query_embeddings = embedding_model.embed_documents(queries, prompt_name="Retrieval-query")
    else:
        query_embeddings = embedding_model.embed_documents(queries)

    eval_metrics = {}

    for strategy in chunking_strat_config:
        print(f"\n=== Strategy: {strategy} ===", flush=True)
        
        chunker, params= chunking_strat_config[strategy]
        chunker = chunker(**params, tokenizer=tokenizer)
        chunked_datas = []
        metadatas = []
        
        start = time.time()
        for doc_idx, doc in enumerate(loaded_docs):
            try:
                chunks = chunker.chunk(dl_doc=doc)
                chunks = [chunker.contextualize(chunk) for chunk in chunks]
                chunked_datas.extend(chunks)
                metadatas.extend([{"chunk_id": f"{doc_idx}_{strategy}_{chunk_idx}"} for chunk_idx in range(len(chunks))])
            except Exception as e:
                print("Chunking went wrong:", str(e))
        chunking_time = time.time() - start
        num_chunks = len(chunked_datas)
        print(f"⏱️ Chunking: {chunking_time:.2f}s | Chunks: {num_chunks}", flush=True)
        
        start = time.time()
        vector_store = FAISS.from_texts(chunked_datas, embedding_model, metadatas=metadatas)
        embedding_faiss_time = time.time() - start
        print(f"⏱️ Embedding + FAISS indexing: {embedding_faiss_time:.2f}s", flush=True)
        
        start = time.time()
        retrieved_docs = []
        for query_idx, query_embedding in enumerate(query_embeddings):
            retrieved_doc = vector_store.similarity_search_by_vector(query_embedding, k=5)#, distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT)
            retrieved_doc = [doc.page_content for doc in retrieved_doc]
            filtered_eval_qa_dataset[query_idx][f"retrieved_docs_{embedding_model_name}_{strategy}"] = retrieved_doc
            retrieved_docs.append(retrieved_doc)
        search_time = time.time() - start
        print(f"⏱️ Search (3 queries): {search_time:.2f}s", flush=True)
        
        start = time.time()
        computed_metrics = retrieval_metrics(retrieved_docs, ground_truth_docs)
        metrics_time = time.time() - start
        print(f"⏱️ Metrics computation: {metrics_time:.2f}s", flush=True)
        
        eval_metrics[strategy] = computed_metrics
        
        runtime_metrics.append({
            'embedding_model': embedding_model_name,
            'strategy': strategy,
            'chunking_time_s': round(chunking_time, 2),
            'num_chunks': num_chunks,
            'embedding_faiss_time_s': round(embedding_faiss_time, 2),
            'search_time_s': round(search_time, 2),
            'metrics_time_s': round(metrics_time, 2),
            'total_embedding_model_time_s': 0
        })

    total_embedding_model_time = time.time() - embedding_model_start_time
    print(f"⏱️ Embedding Model({embedding_model_name}) {total_embedding_model_time:.2f}s", flush=True)
    
    for metric in runtime_metrics:
        if metric['embedding_model'] == embedding_model_name:
            metric['total_embedding_model_time_s'] = round(total_embedding_model_time, 2)

    results_df = pd.DataFrame.from_dict(eval_metrics, orient="index")
    results_df.index.name = "strategy"
    results_df = results_df.reset_index()
    results_df

    results_df.to_csv(f"{embedding_model_name}_chunking_strategies_comparison_docling.csv")
    clear_gpu(embedding_model)

with open("rag_eval_dataset_docling.json", "w") as f:
    json.dump(filtered_eval_qa_dataset, f, indent=4)

runtime_metrics_df = pd.DataFrame(runtime_metrics)
runtime_metrics_df.to_csv("runtime_metrics_docling2.csv", index=False)
