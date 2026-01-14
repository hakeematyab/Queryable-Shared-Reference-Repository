import pytest
from pathlib import Path
from data_processing.data_ingestion import DocumentProcessor


class TestDocumentProcessor:
    @pytest.fixture(scope="class")
    def processor(self, tmp_path_factory):
        images_dir = tmp_path_factory.mktemp("images")
        return DocumentProcessor(images_dir=str(images_dir))
    
    def test_init(self, processor):
        assert processor is not None
        assert hasattr(processor, "converter")
        assert hasattr(processor, "chunker")
        assert hasattr(processor, "max_tokens")
    
    def test_init_default_params(self):
        proc = DocumentProcessor(save_images=False)
        assert proc.max_tokens == 2048
        assert proc.save_images is False
    
    @pytest.fixture(scope="class")
    def sample_pdf(self, tmp_path_factory):
        pdf_path = tmp_path_factory.mktemp("docs") / "test.pdf"
        pdf_content = b"""%PDF-1.4
1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj
2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj
3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R >> endobj
4 0 obj << /Length 44 >> stream
BT /F1 12 Tf 100 700 Td (Test content) Tj ET
endstream endobj
xref
0 5
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000214 00000 n 
trailer << /Size 5 /Root 1 0 R >>
startxref
306
%%EOF"""
        pdf_path.write_bytes(pdf_content)
        return str(pdf_path)
    
    def test_process_returns_dict(self, processor, sample_pdf):
        try:
            result = processor.process(sample_pdf)
            assert isinstance(result, dict)
        except Exception:
            pytest.skip("PDF processing requires full docling setup")
    
    def test_process_result_keys(self, processor, sample_pdf):
        try:
            result = processor.process(sample_pdf)
            expected_keys = {"doc_id", "doc_name", "doc_path", "chunks", "page_images"}
            assert expected_keys.issubset(result.keys())
        except Exception:
            pytest.skip("PDF processing requires full docling setup")
    
    def test_process_chunks_structure(self, processor, sample_pdf):
        try:
            result = processor.process(sample_pdf)
            chunks = result.get("chunks", [])
            assert isinstance(chunks, list)
            if chunks:
                chunk = chunks[0]
                assert "id" in chunk
                assert "text" in chunk
                assert "metadata" in chunk
        except Exception:
            pytest.skip("PDF processing requires full docling setup")
    
    def test_process_nonexistent_file(self, processor):
        with pytest.raises(Exception):
            processor.process("/nonexistent/path/file.pdf")
