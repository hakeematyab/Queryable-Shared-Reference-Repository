
import os

cache_dir ='/scratch/hakeem.at/Queryable-Shared-Reference-Repository/notebooks/pretrained_models'

os.environ['HF_HOME'] = cache_dir
os.environ['TRANSFORMERS_CACHE'] = cache_dir
os.environ['HUGGINGFACE_HUB_CACHE'] = cache_dir


import os
from collections import Counter
from pathlib import Path
import pandas as pd
import json
import random
import re
from tqdm.auto import tqdm
from pydantic import BaseModel, Field

from langchain_community.document_loaders import PyPDFLoader

import torch
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams
from multiprocessing import Pool, cpu_count


seed = 42
random.seed(seed)

raw_data_path = '../raw_data/'
raw_data_dir_names = ['ethan', 'sai']
raw_data_dirs = [raw_data_path+raw_data_dir_name for raw_data_dir_name in raw_data_dir_names]


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


documents_summary_df = pd.DataFrame(counter.items(), columns=["Document Format", "Document Count"])
documents_summary_df


valid_extensions = ['.html','.pdf','.docx']
metadata_extension = ['.ris']
total_documents = documents_summary_df[documents_summary_df["Document Format"].isin(valid_extensions)]["Document Count"].sum()

print(f"Total Document Count: {total_documents}")



pdf_paths = paths['.pdf']
random.shuffle(pdf_paths)

def pdf2text(pdf_path):
    try:
        pdf_loader = PyPDFLoader(pdf_path)
        docs = pdf_loader.load()
        return docs
    except:
        return []

processes = min(1, cpu_count()-1)

with Pool(processes=processes) as pool:
    list_of_docs = list(tqdm(pool.imap(pdf2text, pdf_paths), total=len(pdf_paths)))
loaded_docs = [each_doc for docs in list_of_docs for each_doc in docs]


top_section_limit = 1000 # num chars
num_citations_limit = 10
def is_citation(page_content):
    page_content = page_content.lower()

    reference_indicators = [
        'references',
        'bibliography',
        'works cited',
        'literature cited',
        'citations'
    ]

    top_section = page_content[:top_section_limit]
    if any(indicator in top_section for indicator in reference_indicators):
        return True
    citation_patterns = [
        r'\[\d+\]',  # [1], [2], etc.
        r'\bet al\.',  # et al.
        r'doi:',
        r'arxiv:',
        r'\(\d{4}\)',  # (2023), (2024)
        r'pp\.\s*\d+',  # pp. 123
        r'vol\.\s*\d+',  # vol. 5
    ]
    citation_count = sum(len(re.findall(pattern, page_content, re.IGNORECASE)) for pattern in citation_patterns)

    if citation_count>num_citations_limit:
        return True

    return False

filtered_docs = [doc for doc in loaded_docs if not is_citation(doc.page_content)]


random.seed(seed)
num_questions = 3000
random.shuffle(filtered_docs)
eval_docs = filtered_docs[:num_questions]


def clean_docs(docs):
    for doc in docs:
        page_content = doc.page_content
        page_content = re.sub('\s+', ' ',page_content).strip()
        doc.page_content = page_content
    return docs
eval_docs = clean_docs(eval_docs)


model_id = "nvidia/Llama-3.3-70B-Instruct-FP8"
gpu_memory_utilization=0.95
max_model_len=4096
max_num_seqs=64
enforce_eager=True
guided_decoding_backend = "outlines"

model = LLM(
    model = model_id,
    gpu_memory_utilization=gpu_memory_utilization,
    max_model_len=max_model_len,
    max_num_seqs=max_num_seqs,
    enforce_eager=enforce_eager,
    guided_decoding_backend=guided_decoding_backend,
)


class SyntheticExample(BaseModel):
    question: str = Field(description="A question based on only the provided document")
    excerpt_paragraph: str = Field(description="Exact text paragraph from the document used to answer the question")
    true_answer: str = Field(description="The correct answer for the question")
    wrong_answer: str = Field(description="An incorrect answer that closely resembles the correct answer but is unambiguously wrong")
    

schema = SyntheticExample.model_json_schema()

max_new_tokens=4096
do_sample=True
temperature=0.7
top_p = 0.95

n=1 

guided_params = GuidedDecodingParams(json=schema)
sampling_params = SamplingParams(
                    temperature=temperature,
                    max_tokens = max_new_tokens,
                    guided_decoding=guided_params,
                    n=n,
)


prompt_template = """You are generating evaluation questions from a document.

Generate a factual question about this document and identify the exact text paragraph that answers it. Then generate the true answer for the question based on the provided document and a wrong answer that closely resembles the correct answer but is factually incorrect.

Requirements:
1. Question must be specific and answerable ONLY from the provided text
2. The paragraphs must be EXACT word-for-word copies from the document. No summarization or paraphrasing
3. Avoid questions with the word "and" unless it's part of a proper noun
4. True answer must be factually correct ONLY based on the provided document
5. Wrong answer must closely resemble the correct answer but be factually incorrect. You may fabricate information not present in the document. The wrong answer must be unambiguously and demonstrably incorrect with no room for interpretation.

Document:
{document}
"""


dataset = []
save_path = "hallucination_eval_dataset.json"
batch_size = 128

for idx, step in enumerate(tqdm(range(0,len(eval_docs), batch_size))):
    print('Batch: ', idx)
    current_docs = eval_docs[step:min(step+batch_size, len(eval_docs))]
    sources = []
    prompts = []
    for page in current_docs:
        source = page.metadata['source']
        prompts.append(prompt_template.format(document = page.page_content))
        sources.append(source)
    try:
        responses = model.generate(
            prompts,
            sampling_params,
        )
        for response, source in zip(responses, sources):
            for output in response.outputs:
                try:
                    data_point = json.loads(output.text)
                    data_point['source'] = source
                    dataset.append(data_point)
                except:
                    pass
    except Exception as e:
        print("Error:", e, flush=True)
    with open(save_path, "w") as f:
        json.dump(dataset, f, indent=4)