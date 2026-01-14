import pytest
from unittest.mock import MagicMock, patch
from orchestration.llm import OllamaClient
from orchestration.tools import create_tools


class TestOllamaClient:
    def test_init(self):
        with patch("orchestration.llm.ChatOllama") as mock_chat:
            mock_chat.return_value = MagicMock()
            client = OllamaClient()
            assert client is not None
            mock_chat.assert_called_once()
    
    def test_llm_attribute_exists(self):
        with patch("orchestration.llm.ChatOllama") as mock_chat:
            mock_llm = MagicMock()
            mock_chat.return_value = mock_llm
            client = OllamaClient()
            assert hasattr(client, "llm")
            assert client.llm is mock_llm
    
    def test_init_with_custom_params(self):
        with patch("orchestration.llm.ChatOllama") as mock_chat:
            mock_chat.return_value = MagicMock()
            OllamaClient(
                model="llama2",
                base_url="http://custom:11434",
                temperature=0.5,
                max_tokens=1024
            )
            mock_chat.assert_called_once_with(
                model="llama2",
                base_url="http://custom:11434",
                temperature=0.5,
                max_tokens=1024
            )


class TestCreateTools:
    def test_returns_list(self):
        mock_searcher = MagicMock()
        tools = create_tools(mock_searcher)
        assert isinstance(tools, list)
        assert len(tools) > 0
    
    def test_tool_has_name(self):
        mock_searcher = MagicMock()
        tools = create_tools(mock_searcher)
        assert tools[0].name == "hybrid_search"
    
    def test_tool_has_description(self):
        mock_searcher = MagicMock()
        tools = create_tools(mock_searcher)
        assert tools[0].description is not None
        assert len(tools[0].description) > 0
    
    def test_tool_with_lock(self):
        mock_searcher = MagicMock()
        mock_lock = MagicMock()
        tools = create_tools(mock_searcher, lock=mock_lock)
        assert len(tools) > 0


class TestGetRagGraph:
    def test_graph_compiles(self):
        from orchestration.rag import get_rag_graph
        
        mock_llm = MagicMock()
        mock_llm.bind_tools = MagicMock(return_value=mock_llm)
        mock_detector = MagicMock()
        mock_tools = []
        
        graph = get_rag_graph(
            system_prompt="Test prompt",
            llm=mock_llm,
            tools=mock_tools,
            hallucination_detector=mock_detector
        )
        
        assert graph is not None
