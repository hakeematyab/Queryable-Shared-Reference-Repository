import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

import aiohttp


BASE_URL = "http://localhost:8000"
EVAL_DATASET = Path(__file__).parent.parent / "data/eval/questions/eval_dataset.json"
PAPERS_DIR = Path(__file__).parent.parent / "data/eval/papers"
RESULTS_DIR = Path(__file__).parent.parent / "data/eval/results"


async def index_papers(session: aiohttp.ClientSession) -> bool:
    papers = [str(p.absolute()) for p in PAPERS_DIR.glob("*.pdf")]
    if not papers:
        print("No papers found to index")
        return False
    
    print(f"Indexing {len(papers)} papers...")
    
    async with session.post(f"{BASE_URL}/index", json={"paper_paths": papers}) as resp:
        async for line in resp.content:
            text = line.decode().strip()
            if text.startswith("data: "):
                data = json.loads(text[6:])
                if data.get("status") == "progress":
                    print(f"  [{data['current']}/{data['total']}] {data['file']}")
                elif data.get("status") == "done":
                    print(f"Indexed {data['indexed']} papers, {data['failed']} failed")
                    return data["failed"] == 0
    return False


async def measure_chat_latency(session: aiohttp.ClientSession, question: str, thread_id: str) -> float:
    start = time.perf_counter()
    
    async with session.post(f"{BASE_URL}/chat", json={"user_query": question, "thread_id": thread_id}) as resp:
        async for line in resp.content:
            text = line.decode().strip()
            if text.startswith("data: "):
                data = json.loads(text[6:])
                if data.get("done") or data.get("error"):
                    break
    
    return time.perf_counter() - start


def calculate_percentile(data: list[float], p: int) -> float:
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * p / 100
    f = int(k)
    c = f + 1 if f + 1 < len(sorted_data) else f
    return sorted_data[f] + (sorted_data[c] - sorted_data[f]) * (k - f)


async def run_latency_test(questions: list[dict]) -> dict:
    latencies = []
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(f"{BASE_URL}/health") as resp:
                if resp.status != 200:
                    print("Server not healthy")
                    return {}
        except aiohttp.ClientError as e:
            print(f"Cannot connect to server: {e}")
            return {}
        
        total = len(questions)
        for i, q in enumerate(questions, 1):
            question_text = q["question"]
            thread_id = f"latency_test_{q['id']}"
            
            print(f"[{i}/{total}] {q['id']}: {question_text[:50]}...")
            
            latency = await measure_chat_latency(session, question_text, thread_id)
            latencies.append(latency)
            print(f"         -> {latency:.2f}s")
    
    if not latencies:
        return {}
    
    return {
        "total_questions": len(latencies),
        "min_latency": min(latencies),
        "max_latency": max(latencies),
        "mean_latency": statistics.mean(latencies),
        "median_latency": statistics.median(latencies),
        "p50": calculate_percentile(latencies, 50),
        "p90": calculate_percentile(latencies, 90),
        "p95": calculate_percentile(latencies, 95),
        "p99": calculate_percentile(latencies, 99),
        "all_latencies": latencies
    }


async def main():
    parser = argparse.ArgumentParser(description="P95 Latency Test")
    parser.add_argument("--index", action="store_true", help="Index papers before testing")
    parser.add_argument("--base-url", default=BASE_URL, help="Server base URL")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of questions (0 = all)")
    args = parser.parse_args()
    
    global BASE_URL
    BASE_URL = args.base_url
    
    with open(EVAL_DATASET) as f:
        dataset = json.load(f)
    questions = dataset["questions"]
    if args.limit > 0:
        questions = questions[:args.limit]
    
    print(f"Loaded {len(questions)} questions")
    
    async with aiohttp.ClientSession() as session:
        if args.index:
            if not await index_papers(session):
                print("Indexing failed")
    
    print("\n=== Starting P95 Latency Test ===\n")
    results = await run_latency_test(questions)
    
    if not results:
        print("No results collected")
        return
    
    print("\n=== Results ===")
    print(f"Total Questions: {results['total_questions']}")
    print(f"Min Latency:     {results['min_latency']:.2f}s")
    print(f"Max Latency:     {results['max_latency']:.2f}s")
    print(f"Mean Latency:    {results['mean_latency']:.2f}s")
    print(f"Median Latency:  {results['median_latency']:.2f}s")
    print(f"P50:             {results['p50']:.2f}s")
    print(f"P90:             {results['p90']:.2f}s")
    print(f"P95:             {results['p95']:.2f}s")
    print(f"P99:             {results['p99']:.2f}s")
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_file = RESULTS_DIR / "latency_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    asyncio.run(main())
