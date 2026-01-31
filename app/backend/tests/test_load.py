import argparse
import asyncio
import json
import time
from pathlib import Path

import aiohttp


BASE_URL = "http://localhost:8000"
EVAL_DATASET = Path(__file__).parent.parent / "data/eval/questions/eval_dataset.json"
PAPERS_DIR = Path(__file__).parent.parent / "data/eval/papers"
RESULTS_DIR = Path(__file__).parent.parent / "data/eval/results"
CONCURRENCY_LEVELS = [1, 5, 10]
QUESTIONS_PER_LEVEL = 10


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


async def send_request(session: aiohttp.ClientSession, question: str, thread_id: str) -> dict:
    start = time.perf_counter()
    success = False
    error = None
    
    try:
        async with session.post(
            f"{BASE_URL}/chat",
            json={"user_query": question, "thread_id": thread_id},
            timeout=aiohttp.ClientTimeout(total=300)
        ) as resp:
            async for line in resp.content:
                text = line.decode().strip()
                if text.startswith("data: "):
                    data = json.loads(text[6:])
                    if data.get("done"):
                        success = True
                        break
                    if data.get("error"):
                        error = data["error"]
                        break
    except Exception as e:
        error = str(e)
    
    elapsed = time.perf_counter() - start
    return {"latency": elapsed, "success": success, "error": error}


async def run_concurrent_test(session: aiohttp.ClientSession, questions: list[dict], concurrency: int) -> dict:
    tasks = []
    
    for i, q in enumerate(questions):
        thread_id = f"load_test_c{concurrency}_{q['id']}"
        task = send_request(session, q["question"], thread_id)
        tasks.append(task)
    
    start = time.perf_counter()
    results = await asyncio.gather(*tasks)
    total_time = time.perf_counter() - start
    
    latencies = [r["latency"] for r in results]
    successes = sum(1 for r in results if r["success"])
    
    return {
        "concurrency": concurrency,
        "total_requests": len(results),
        "successful": successes,
        "failed": len(results) - successes,
        "success_rate": successes / len(results) * 100 if results else 0,
        "total_time": total_time,
        "throughput": len(results) / total_time if total_time > 0 else 0,
        "avg_latency": sum(latencies) / len(latencies) if latencies else 0,
        "min_latency": min(latencies) if latencies else 0,
        "max_latency": max(latencies) if latencies else 0,
    }


async def main():
    parser = argparse.ArgumentParser(description="Load Test")
    parser.add_argument("--index", action="store_true", help="Index papers before testing")
    parser.add_argument("--base-url", default=BASE_URL, help="Server base URL")
    parser.add_argument("--questions-per-level", type=int, default=QUESTIONS_PER_LEVEL, help="Questions per concurrency level")
    args = parser.parse_args()
    
    global BASE_URL
    BASE_URL = args.base_url
    
    with open(EVAL_DATASET) as f:
        dataset = json.load(f)
    all_questions = dataset["questions"]
    
    print(f"Loaded {len(all_questions)} questions")
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(f"{BASE_URL}/health") as resp:
                if resp.status != 200:
                    print("Server not healthy")
                    return
        except aiohttp.ClientError as e:
            print(f"Cannot connect to server: {e}")
            return
        
        if args.index:
            if not await index_papers(session):
                print("Indexing failed")
        
        print("\n=== Starting Load Test ===\n")
        all_results = []
        
        question_offset = 0
        for concurrency in CONCURRENCY_LEVELS:
            questions = all_questions[question_offset:question_offset + args.questions_per_level]
            question_offset += args.questions_per_level
            
            if not questions:
                print(f"Not enough questions for concurrency {concurrency}")
                break
            
            print(f"Testing concurrency={concurrency} with {len(questions)} questions...")
            
            result = await run_concurrent_test(session, questions, concurrency)
            all_results.append(result)
            
            print(f"  Success Rate:  {result['success_rate']:.1f}%")
            print(f"  Throughput:    {result['throughput']:.2f} req/s")
            print(f"  Avg Latency:   {result['avg_latency']:.2f}s")
            print(f"  Min/Max:       {result['min_latency']:.2f}s / {result['max_latency']:.2f}s")
            print()
    
    print("=== Summary ===")
    print(f"{'Concurrency':<12} {'Success%':<10} {'Throughput':<12} {'Avg Latency':<12}")
    print("-" * 50)
    for r in all_results:
        print(f"{r['concurrency']:<12} {r['success_rate']:<10.1f} {r['throughput']:<12.2f} {r['avg_latency']:<12.2f}")
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_file = RESULTS_DIR / "load_results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    asyncio.run(main())
