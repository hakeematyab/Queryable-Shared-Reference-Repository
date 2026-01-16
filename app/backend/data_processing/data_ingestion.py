import os
import sys
import logging
import warnings
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from contextlib import contextmanager

warnings.filterwarnings("ignore")
for name in ["docling", "docling_parse", "PIL", "pdfplumber", "pdfminer"]:
    logging.getLogger(name).setLevel(logging.ERROR)

from docling.document_converter import (
    DocumentConverter, 
    PdfFormatOption, 
    WordFormatOption, 
    HTMLFormatOption
)
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
from docling.backend.docling_parse_v2_backend import DoclingParseV2DocumentBackend
from docling.chunking import HybridChunker

logger = logging.getLogger(__name__)


@contextmanager
def suppress_output():
    """Suppress stdout/stderr during noisy operations."""
    with open(os.devnull, 'w') as null:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = null, null
        try:
            yield
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr


class DocumentProcessor:
    
    def __init__(
        self,
        max_tokens: int = 512,
        images_dir: Optional[str] = "./data/images",
        save_images: bool = True,
        image_quality: int = 70,
    ):
        self.max_tokens = max_tokens
        self.images_dir = Path(images_dir) if images_dir else None
        self.save_images = save_images
        self.image_quality = image_quality
        
        if self.images_dir:
            self.images_dir.mkdir(parents=True, exist_ok=True)
        
        pdf_options = PdfPipelineOptions()
        pdf_options.do_ocr = False
        pdf_options.images_scale = 2.0
        pdf_options.generate_page_images = save_images
        pdf_options.generate_picture_images = False
        
        self.converter = DocumentConverter(
            allowed_formats={InputFormat.PDF, InputFormat.DOCX, InputFormat.HTML},
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pdf_options,
                    pipeline_cls=StandardPdfPipeline,
                    backend=DoclingParseV2DocumentBackend
                ),
                InputFormat.DOCX: WordFormatOption(),
                InputFormat.HTML: HTMLFormatOption(),
            }
        )
        
        self.chunker = HybridChunker(merge_peers=True, max_tokens=max_tokens)
        logger.info(f"DocumentProcessor initialized (max_tokens={max_tokens})")
    
    def process(self, doc_path: str) -> Dict:
        doc_path = Path(doc_path)
        ext = doc_path.suffix.lower()
        
        with suppress_output():
            conv_result = self.converter.convert(source=str(doc_path))
        
        doc = conv_result.document
        doc_id = self._get_doc_hash(conv_result)
        doc_name = doc.origin.filename if doc.origin else doc_path.name
        
        page_images = {}
        if ext == '.pdf' and self.save_images and self.images_dir:
            page_images = self._save_page_images(conv_result, doc_id)
        
        chunks_raw = list(self.chunker.chunk(dl_doc=doc))
        
        chunks = []
        for idx, chunk in enumerate(chunks_raw):
            chunk_data = self._extract_chunk_data(
                chunk, doc_id, idx, doc_name, str(doc_path)
            )
            chunks.append(chunk_data)
        
        logger.info(f"Processed {doc_name}: {len(chunks)} chunks")
        
        return {
            "doc_id": doc_id,
            "doc_name": doc_name,
            "doc_path": str(doc_path),
            "chunks": chunks,
            "page_images": page_images,
        }
    
    def process_batch(self, doc_paths: List[str]) -> List[Dict]:
        results = []
        for path in doc_paths:
            try:
                result = self.process(path)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process {path}: {e}")
        return results
    
    def _get_doc_hash(self, conv_result) -> str:
        if hasattr(conv_result, 'input') and hasattr(conv_result.input, 'document_hash'):
            return str(conv_result.input.document_hash)
        if hasattr(conv_result.document, 'origin') and conv_result.document.origin:
            if hasattr(conv_result.document.origin, 'binary_hash'):
                return str(conv_result.document.origin.binary_hash)
        return Path(conv_result.input.file).stem
    
    def _extract_chunk_data(
        self, 
        chunk, 
        doc_id: str, 
        chunk_idx: int, 
        doc_name: str,
        doc_path: str,
    ) -> Dict:
        page_numbers = set()
        bboxes = []
        
        if hasattr(chunk, 'meta') and hasattr(chunk.meta, 'doc_items'):
            for doc_item in chunk.meta.doc_items:
                if hasattr(doc_item, 'prov') and doc_item.prov:
                    for prov in doc_item.prov:
                        if hasattr(prov, 'page_no'):
                            page_numbers.add(prov.page_no)
                        if hasattr(prov, 'bbox'):
                            bboxes.append({
                                'page_no': prov.page_no if hasattr(prov, 'page_no') else None,
                                'l': prov.bbox.l,
                                't': prov.bbox.t,
                                'r': prov.bbox.r,
                                'b': prov.bbox.b,
                                'coord_origin': str(prov.bbox.coord_origin) if hasattr(prov.bbox, 'coord_origin') else 'BOTTOMLEFT'
                            })
        
        headings = []
        if hasattr(chunk, 'meta') and hasattr(chunk.meta, 'headings'):
            headings = list(chunk.meta.headings) if chunk.meta.headings else []
        
        chunk_id = f"{doc_id}_chunk_{chunk_idx:04d}"
        
        import json
        return {
            "id": chunk_id,
            "text": self.chunker.serialize(chunk),
            "metadata": {
                "doc_id": doc_id,
                "doc_name": doc_name,
                "doc_path": doc_path,
                "chunk_index": chunk_idx,
                "page_numbers": json.dumps(sorted(list(page_numbers))),
                "bboxes": json.dumps(bboxes),
                "headings": json.dumps(headings),
                "raw_text": chunk.text,
            }
        }
    
    def _save_page_images(self, conv_result, doc_id: str) -> Dict[int, str]:
        page_paths = {}
        if not hasattr(conv_result.document, 'pages'):
            return page_paths
        
        for page_no, page in conv_result.document.pages.items():
            if hasattr(page, 'image') and page.image and hasattr(page.image, 'pil_image'):
                img = page.image.pil_image
                if img.mode in ('RGBA', 'P'):
                    img = img.convert('RGB')
                
                img_path = self.images_dir / f"{doc_id}_page_{page_no}.jpg"
                img.save(img_path, format='JPEG', quality=self.image_quality, optimize=True)
                page_paths[page_no] = str(img_path)
        
        return page_paths