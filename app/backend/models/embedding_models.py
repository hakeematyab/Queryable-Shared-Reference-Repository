import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger(__name__)


class EmbeddingGemmaEmbeddings:
    def __init__(self, model_id='google/embeddinggemma-300m', device="cpu", max_seq_length=2048, batch_size=32):
        self.model = SentenceTransformer(model_id, device=device, trust_remote_code=True)
        self.model.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.tokenizer = self.model.tokenizer
        self._dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f"Loaded EmbeddingGemma on {device}, dim={self._dimension}")
    
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


class MiniLMEmbeddings:
    def __init__(self, model_id='sentence-transformers/all-MiniLM-L6-v2', device="cpu", max_length = 512, batch_size=32, normalize=True):
        self.batch_size = batch_size
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id)
        self.normalize = normalize
        self.device = device
        self.model.to(device)
        self._dimension = self.model.config.hidden_size
        logger.info(f"Loaded MiniLM on {device}, dim={self._dimension}")
    
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
                                       return_tensors="pt", max_length=self.max_length).to(self.device)
                output = self.model(**inputs)
                embeddings = self._mean_pooling(output, inputs['attention_mask'])
                if self.normalize:
                    embeddings = F.normalize(embeddings, p=2, dim=1)
                outputs.extend(embeddings.cpu().tolist())
        return outputs
    
    def embed_query(self, text):
        self.model.eval()
        inputs = self.tokenizer([text], padding=True, truncation=True, 
                               return_tensors="pt", max_length=self.max_length).to(self.device)
        with torch.no_grad():
            output = self.model(**inputs)
            embedding = self._mean_pooling(output, inputs['attention_mask'])
            if self.normalize:
                embedding = F.normalize(embedding, p=2, dim=1)
        return embedding.cpu().tolist()[0]
    



class MPNetEmbeddings:
    def __init__(self, model_id='sentence-transformers/all-mpnet-base-v2', device="cpu", batch_size=32, max_length = 512, normalize=True):
        self.batch_size = batch_size
        self.max_length = max_length
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id)
        self.normalize = normalize
        self.device = device
        self.model.to(device)
        self._dimension = self.model.config.hidden_size
        logger.info(f"Loaded MPNet on {device}, dim={self._dimension}")
    
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
                                       return_tensors="pt", max_length=self.max_length).to(self.device)
                output = self.model(**inputs)
                embeddings = self._mean_pooling(output, inputs['attention_mask'])
                if self.normalize:
                    embeddings = F.normalize(embeddings, p=2, dim=1)
                outputs.extend(embeddings.cpu().tolist())
        return outputs
    
    def embed_query(self, text):
        self.model.eval()
        inputs = self.tokenizer([text], padding=True, truncation=True, 
                               return_tensors="pt", max_length=self.max_length).to(self.device)
        with torch.no_grad():
            output = self.model(**inputs)
            embedding = self._mean_pooling(output, inputs['attention_mask'])
            if self.normalize:
                embedding = F.normalize(embedding, p=2, dim=1)
        return embedding.cpu().tolist()[0]

class SciNCLEmbeddings:
    def __init__(self, model_id='malteos/scincl', device="cpu", max_length = 512, batch_size=32):
        self.batch_size = batch_size
        self.max_length = max_length
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id)
        self.device = device
        self.model.to(device)
        self._dimension = self.model.config.hidden_size
        logger.info(f"Loaded SciNCL on {device}, dim={self._dimension}")
    
    def embed_documents(self, texts):
        outputs = []
        self.model.eval()
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:min(i + self.batch_size, len(texts))]
            with torch.no_grad():
                inputs = self.tokenizer(batch, padding=True, truncation=True, 
                                       return_tensors="pt", max_length=self.max_length).to(self.device)
                output = self.model(**inputs)
                embeddings = output.last_hidden_state[:, 0, :].cpu().tolist()
                outputs.extend(embeddings)
        return outputs
    
    def embed_query(self, text):
        self.model.eval()
        inputs = self.tokenizer([text], padding=True, truncation=True, 
                               return_tensors="pt", max_length=self.max_length).to(self.device)
        with torch.no_grad():
            output = self.model(**inputs)
            embedding = output.last_hidden_state[:, 0, :].cpu().tolist()[0]
        return embedding


EMBEDDING_MODELS = {
    "gemma": EmbeddingGemmaEmbeddings,
    "minilm": MiniLMEmbeddings,
    "mpnet": MPNetEmbeddings,
    "scincl": SciNCLEmbeddings,
}