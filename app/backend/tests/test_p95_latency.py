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


async def measure_chat_latency(session: aiohttp.ClientSession, question: str, thread_id: str) -> dict:
    start = time.perf_counter()
    ttft = None
    
    async with session.post(f"{BASE_URL}/chat", json={"user_query": question, "thread_id": thread_id}) as resp:
        async for line in resp.content:
            text = line.decode().strip()
            if text.startswith("data: "):
                data = json.loads(text[6:])
                if data.get("token") and ttft is None:
                    ttft = time.perf_counter() - start
                if data.get("done") or data.get("error"):
                    break
    
    return {"ttft": ttft, "total": time.perf_counter() - start}


def calculate_percentile(data: list[float], p: int) -> float:
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * p / 100
    f = int(k)
    c = f + 1 if f + 1 < len(sorted_data) else f
    return sorted_data[f] + (sorted_data[c] - sorted_data[f]) * (k - f)


async def run_latency_test(questions: list[dict]) -> dict:
    ttft_list = []
    total_list = []
    
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
            
            result = await measure_chat_latency(session, question_text, thread_id)
            if result["ttft"] is not None:
                ttft_list.append(result["ttft"])
            total_list.append(result["total"])
            ttft_str = f"{result['ttft']:.2f}s" if result["ttft"] else "N/A"
            print(f"         -> TTFT: {ttft_str}, Total: {result['total']:.2f}s")
    
    if not total_list:
        return {}
    
    return {
        "total_questions": len(total_list),
        "ttft": {
            "min": min(ttft_list) if ttft_list else None,
            "max": max(ttft_list) if ttft_list else None,
            "mean": statistics.mean(ttft_list) if ttft_list else None,
            "p50": calculate_percentile(ttft_list, 50) if ttft_list else None,
            "p90": calculate_percentile(ttft_list, 90) if ttft_list else None,
            "p95": calculate_percentile(ttft_list, 95) if ttft_list else None,
            "p99": calculate_percentile(ttft_list, 99) if ttft_list else None,
        },
        "total": {
            "min": min(total_list),
            "max": max(total_list),
            "mean": statistics.mean(total_list),
            "p50": calculate_percentile(total_list, 50),
            "p90": calculate_percentile(total_list, 90),
            "p95": calculate_percentile(total_list, 95),
            "p99": calculate_percentile(total_list, 99),
        },
        "all_ttft": ttft_list,
        "all_total": total_list
    }


async def main():
    global BASE_URL

    parser = argparse.ArgumentParser(description="P95 Latency Test")
    parser.add_argument("--index", action="store_true", help="Index papers before testing")
    parser.add_argument("--base-url", default=BASE_URL, help="Server base URL")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of questions (0 = all)")
    args = parser.parse_args()
    
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
    
    ttft = results["ttft"]
    total = results["total"]
    
    print("\n=== Time to First Token (TTFT) ===")
    if ttft["p95"]:
        print(f"Min:  {ttft['min']:.2f}s | Max: {ttft['max']:.2f}s | Mean: {ttft['mean']:.2f}s")
        print(f"P50:  {ttft['p50']:.2f}s | P90: {ttft['p90']:.2f}s | P95: {ttft['p95']:.2f}s | P99: {ttft['p99']:.2f}s")
    else:
        print("No TTFT data")
    
    print("\n=== Total Response Time ===")
    print(f"Min:  {total['min']:.2f}s | Max: {total['max']:.2f}s | Mean: {total['mean']:.2f}s")
    print(f"P50:  {total['p50']:.2f}s | P90: {total['p90']:.2f}s | P95: {total['p95']:.2f}s | P99: {total['p99']:.2f}s")
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_file = RESULTS_DIR / "latency_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    asyncio.run(main())
