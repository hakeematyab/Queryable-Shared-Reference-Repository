import json
import logging
import threading
import time
import uuid
from pathlib import Path
from contextlib import asynccontextmanager
from typing import List
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from filelock import FileLock, Timeout

from models.embedding import EmbeddingModel
from models.reranker import RerankerModel
from models.hallucination import HallucinationDetector
from retrieval.vector_store import VectorStore
from retrieval.retrieval import HybridSearcher
from data_processing.data_ingestion import DocumentProcessor
from orchestration.llm import OllamaClient
from utils.logging_config import setup_session_logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

session_log_path = setup_session_logging(user_id="server")
logger.info(f"Session log file: {session_log_path}")

INDEX_LOCK_PATH = "./data/.index.lock"
CHECKPOINT_DB = "./data/checkpoints.db"
UPLOAD_DIR = Path("./uploads")

app_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    from orchestration.tools import create_tools
    from orchestration.prompts import SYSTEM_PROMPT
    from orchestration.rag import get_rag_graph
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    
    logger.info("Starting RAG server initialization")
    
    logger.info("Initializing embedding model")
    app_state["embedding"] = EmbeddingModel()
    
    logger.info("Initializing reranker model")
    app_state["reranker"] = RerankerModel()
    
    logger.info("Initializing hallucination detector")
    app_state["hallucination"] = HallucinationDetector()
    
    logger.info("Initializing Ollama LLM client")
    app_state["llm"] = OllamaClient(
        model="qwen3:8b",
        base_url="http://localhost:11434",
        temperature=0.7,
        max_tokens=2048,
    )
    
    logger.info("Initializing vector store")
    app_state["vector_store"] = VectorStore()
    app_state["vector_store"].load()
    app_state["vector_store_lock"] = threading.Lock()
    
    logger.info("Initializing hybrid searcher")
    app_state["searcher"] = HybridSearcher(
        app_state["vector_store"], 
        app_state["embedding"], 
        app_state["reranker"]
    )
    
    logger.info("Creating RAG tools")
    tools = create_tools(app_state["searcher"], app_state["vector_store_lock"])
    
    logger.info("Setting up async SQLite checkpointer")
    Path("./data").mkdir(exist_ok=True)
    
    async with AsyncSqliteSaver.from_conn_string(CHECKPOINT_DB) as checkpointer:
        await checkpointer.setup()
        app_state["checkpointer"] = checkpointer
        logger.info("Checkpointer initialized at %s", CHECKPOINT_DB)
        
        logger.info("Building RAG graph")
        app_state["graph"] = get_rag_graph(
            SYSTEM_PROMPT, 
            app_state["llm"].llm, 
            tools, 
            app_state["hallucination"],
            checkpointer=checkpointer
        )
        
        logger.info("Server initialization complete. Vector store count: %s", app_state["vector_store"].count())
        
        yield

app = FastAPI(lifespan=lifespan, title="RAG Pipeline Server")
app.add_middleware(
    CORSMiddleware, 
    allow_origins=["*"], 
    allow_credentials=True, 
    allow_methods=["*"], 
    allow_headers=["*"]
)

class ChatRequest(BaseModel):
    user_query: str
    thread_id: str = "default"

class IndexRequest(BaseModel):
    paper_paths: list[str]

@app.get("/health")
async def health():
    counts = app_state["vector_store"].count() if "vector_store" in app_state else None
    logger.info(f"[ENDPOINT: /health] Status: healthy, Vector store: {counts}")
    return {"status": "healthy", "vector_store": counts}

@app.get("/index/status")
async def index_status():
    lock = FileLock(INDEX_LOCK_PATH, timeout=0)
    try:
        lock.acquire()
        lock.release()
        return {"indexing": False}
    except Timeout:
        return {"indexing": True}

@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    logger.info(f"[ENDPOINT: /upload] Received {len(files)} files")
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    
    saved_paths, errors = [], []
    for file in files:
        try:
            safe_filename = f"{uuid.uuid4().hex[:8]}_{file.filename}"
            file_path = UPLOAD_DIR / safe_filename
            content = await file.read()
            with open(file_path, "wb") as f:
                f.write(content)
            saved_paths.append(str(file_path.absolute()))
            logger.info(f"[ENDPOINT: /upload] Saved: {safe_filename}")
        except Exception as e:
            logger.error(f"[ENDPOINT: /upload] Failed: {file.filename}: {e}")
            errors.append({"filename": file.filename, "error": str(e)})
    
    return {"saved_paths": saved_paths, "saved_count": len(saved_paths), "errors": errors}

@app.get("/chats/{user_hash}")
async def get_chats(user_hash: str, limit: int = 10, offset: int = 0):
    logger.info(f"[ENDPOINT: /chats] User: {user_hash}, Limit: {limit}, Offset: {offset}")
    try:
        checkpointer = app_state.get("checkpointer")
        if not checkpointer:
            raise HTTPException(500, "Checkpointer not initialized")
        
        import aiosqlite
        
        async with aiosqlite.connect(CHECKPOINT_DB) as db:
            cursor = await db.execute("""
                SELECT DISTINCT thread_id 
                FROM checkpoints 
                WHERE thread_id LIKE ?
                ORDER BY thread_id DESC
                LIMIT ? OFFSET ?
            """, (f"{user_hash}_%", limit + 1, offset))
            
            rows = await cursor.fetchall()
            has_more = len(rows) > limit
            rows = rows[:limit]
        
        threads = []
        for (thread_id,) in rows:
            parts = thread_id.rsplit("_", 1)
            timestamp = int(parts[-1]) if len(parts) >= 2 and parts[-1].isdigit() else None
            
            preview = "New conversation"
            try:
                config = {"configurable": {"thread_id": thread_id}}
                checkpoint_tuple = await checkpointer.aget_tuple(config)
                if checkpoint_tuple and checkpoint_tuple.checkpoint:
                    channel_values = checkpoint_tuple.checkpoint.get("channel_values", {})
                    raw_messages = channel_values.get("messages", [])
                    for msg in raw_messages:
                        content = None
                        if hasattr(msg, "type") and msg.type == "human":
                            content = msg.content
                        elif isinstance(msg, dict) and msg.get("type") == "human":
                            content = msg.get("content", "")
                        if content:
                            preview = content[:100] + "..." if len(content) > 100 else content
                            break
            except Exception as e:
                logger.warning("Failed to get preview for thread %s: %s", thread_id, e)
            
            threads.append({
                "thread_id": thread_id,
                "created_at": timestamp,
                "preview": preview
            })
        
        logger.info(f"[ENDPOINT: /chats] Returning {len(threads)} threads, has_more: {has_more}")
        return {"threads": threads, "offset": offset, "has_more": has_more}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to retrieve chats for user %s: %s", user_hash, e)
        raise HTTPException(500, f"Database error: {str(e)}")

@app.get("/history/{thread_id}")
async def get_history(thread_id: str):
    logger.info(f"[ENDPOINT: /history] Thread: {thread_id}")
    try:
        checkpointer = app_state.get("checkpointer")
        if not checkpointer:
            raise HTTPException(500, "Checkpointer not initialized")
        
        config = {"configurable": {"thread_id": thread_id}}
        checkpoint_tuple = await checkpointer.aget_tuple(config)
        
        if not checkpoint_tuple or not checkpoint_tuple.checkpoint:
            raise HTTPException(404, "Thread not found")
        
        checkpoint_data = checkpoint_tuple.checkpoint
        channel_values = checkpoint_data.get("channel_values", {})
        raw_messages = channel_values.get("full_history", [])
        citations_history = channel_values.get("citations_history", [])
        hallucination_scores = channel_values.get("hallucination_scores", [])
        
        messages = []
        ai_msg_idx = 0
        
        for msg in raw_messages:
            if hasattr(msg, "type") and hasattr(msg, "content"):
                msg_type = msg.type
                content = msg.content
            elif isinstance(msg, dict):
                msg_type = msg.get("type", "unknown")
                content = msg.get("content", "")
            else:
                continue
            
            if msg_type == "human":
                messages.append({"role": "user", "content": content})
            elif msg_type == "ai":
                msg_data = {"role": "assistant", "content": content}
                if ai_msg_idx < len(citations_history):
                    msg_data["citations"] = citations_history[ai_msg_idx]
                if ai_msg_idx < len(hallucination_scores):
                    msg_data["hallucination_score"] = hallucination_scores[ai_msg_idx]
                messages.append(msg_data)
                ai_msg_idx += 1
        
        logger.info(f"[ENDPOINT: /history] Returning {len(messages)} messages for thread {thread_id}")
        return {"thread_id": thread_id, "messages": messages}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to retrieve history for thread %s: %s", thread_id, e)
        raise HTTPException(500, f"Database error: {str(e)}")


@app.post("/chat")
async def chat(request: ChatRequest):
    logger.info(f"[ENDPOINT: /chat] Thread: {request.thread_id}, Query: {request.user_query[:100]}..." if len(request.user_query) > 100 else f"[ENDPOINT: /chat] Thread: {request.thread_id}, Query: {request.user_query}")
    async def generate():
        is_streaming = False
        start_time = time.time()
        first_token_time = None
        
        try:
            state = {"query": request.user_query}
            config = {"configurable": {"thread_id": request.thread_id}}
            
            async for event in app_state["graph"].astream_events(state, config, version="v2"):
                
                event_type = event.get("event")
                metadata = event.get("metadata", {})
                node = metadata.get("langgraph_node") if isinstance(metadata, dict) else None
                event_name = event.get("name", "")
                
                if event_type == "on_chain_end":
                    if node == "data_validation" or event_name == "data_validation":
                        output = event.get("data", {}).get("output", {})
                        if isinstance(output, dict) and not output.get("is_data_valid", True):
                            yield f'data: {json.dumps({"done": False, "token": "Your query is too long. Please shorten it and try again."})}\n\n'
                            yield f'data: {json.dumps({"done": True, "citations": []})}\n\n'
                            return
                    
                    elif node == "denial" or event_name == "denial":
                        output = event.get("data", {}).get("output", {})
                        msg = output.get("response", "Query too long.") if isinstance(output, dict) else "Query too long."
                        yield f'data: {json.dumps({"done": False, "token": msg})}\n\n'
                        yield f'data: {json.dumps({"done": True, "citations": []})}\n\n'
                        return
                
                elif event_type == "on_chain_start":
                    if node == "react_loop" or event_name == "react_loop":
                        is_streaming = True
                
                elif event_type == "on_chain_end":
                    if node == "react_loop" or event_name == "react_loop":
                        is_streaming = False
                
                elif event_type == "on_chat_model_stream" and is_streaming:
                    chunk = event.get("data", {}).get("chunk")
                    if chunk and hasattr(chunk, "content") and chunk.content:
                        if hasattr(chunk, "tool_calls") and chunk.tool_calls:
                            continue
                        # Track time to first token
                        if first_token_time is None:
                            first_token_time = time.time()
                            ttft = first_token_time - start_time
                            logger.info(f"[TTFT] /chat thread_id={request.thread_id} | {ttft:.2f}s")
                        yield f'data: {json.dumps({"done": False, "token": chunk.content})}\n\n'
            
            final_state = await app_state["graph"].aget_state(config)
            state_values = final_state.values if final_state else {}
            
            citations = state_values.get("citations", []) or []
            hallucination_score = state_values.get("hallucination_score")
            
            elapsed_time = time.time() - start_time
            ttft_str = f"{first_token_time - start_time:.2f}s" if first_token_time else "N/A"
            logger.info(f"[RESPONSE_TIME] /chat thread_id={request.thread_id} | TTFT={ttft_str} Total={elapsed_time:.2f}s")
            logger.info(f"[ENDPOINT: /chat] Complete. Citations: {len(citations)}, Hallucination score: {hallucination_score}")
            yield f'data: {json.dumps({"done": True, "citations": citations, "hallucination_score": hallucination_score})}\n\n'
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            logger.error(f"[RESPONSE_TIME] /chat thread_id={request.thread_id} ERROR | {elapsed_time:.2f}s - {str(e)}")
            yield f'data: {json.dumps({"error": str(e)})}\n\n'
    
    return StreamingResponse(
        generate(), 
        media_type="text/event-stream", 
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )

@app.post("/index")
async def index(request: IndexRequest):
    logger.info(f"[ENDPOINT: /index] Received {len(request.paper_paths)} paths")
    valid = [str(Path(p).absolute()) for p in request.paper_paths if Path(p).exists()]
    logger.info(f"[ENDPOINT: /index] Valid paths: {len(valid)}")
    if not valid:
        raise HTTPException(400, "No valid paths")
    
    lock = FileLock(INDEX_LOCK_PATH, timeout=0)
    
    async def stream_progress():
        start_time = time.time()
        try:
            lock.acquire()
        except Timeout:
            logger.warning(f"[RESPONSE_TIME] /index BLOCKED - lock acquisition failed")
            yield f'data: {json.dumps({"status": "error", "error": "Another indexing operation is in progress. Please wait."})}\n\n'
            return
        
        try:
            processor = DocumentProcessor()
            indexed = []
            failed = []
            
            for i, path in enumerate(valid):
                try:
                    result = processor.process(path)
                    ids = [c["id"] for c in result["chunks"]]
                    texts = [c["text"] for c in result["chunks"]]
                    metas = [c["metadata"] for c in result["chunks"]]
                    embeddings = app_state["embedding"].embed_documents_sync(texts)
                    with app_state["vector_store_lock"]:
                        app_state["vector_store"].add(ids, texts, embeddings, metas)
                    indexed.append(path)
                    logger.info(f"[ENDPOINT: /index] Indexed {Path(path).name}: {len(ids)} chunks")
                    yield f'data: {json.dumps({"status": "progress", "current": i+1, "total": len(valid), "file": Path(path).name, "success": True})}\n\n'
                except Exception as e:
                    failed.append({"file": Path(path).name, "error": str(e)})
                    yield f'data: {json.dumps({"status": "progress", "current": i+1, "total": len(valid), "file": Path(path).name, "success": False, "error": str(e)})}\n\n'
            
            with app_state["vector_store_lock"]:
                app_state["vector_store"].save()
            
            elapsed_time = time.time() - start_time
            logger.info(f"[RESPONSE_TIME] /index files={len(valid)} indexed={len(indexed)} failed={len(failed)} | {elapsed_time:.2f}s")
            logger.info(f"[ENDPOINT: /index] Complete. Indexed: {len(indexed)}, Failed: {len(failed)}")
            yield f'data: {json.dumps({"status": "done", "indexed": len(indexed), "failed": len(failed), "failed_files": failed})}\n\n'
        finally:
            lock.release()
    
    return StreamingResponse(
        stream_progress(), 
        media_type="text/event-stream", 
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
