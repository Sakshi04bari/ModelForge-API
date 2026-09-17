import asyncio
import os
import time
import httpx

API_URL = "http://localhost:8000/api/v1/predict"
API_KEY = os.getenv("API_KEY")

if not API_KEY:
    raise RuntimeError("API_KEY environment variable is not set")

payload = {
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
}

headers = {
    "X-API-Key": API_KEY
}


async def send_request(client):
    start = time.perf_counter()

    try:
        response = await client.post(
            API_URL,
            json=payload,
            headers=headers,
            timeout=10
        )

        duration = time.perf_counter() - start

        return response.status_code == 200, duration

    except Exception:
        duration = time.perf_counter() - start
        return False, duration


async def main():
    async with httpx.AsyncClient() as client:
        start_time = time.perf_counter()

        results = await asyncio.gather(
            *[send_request(client) for _ in range(100)]
        )

        total_time = time.perf_counter() - start_time

    successful = sum(success for success, _ in results)
    failed = len(results) - successful

    latencies = [duration for _, duration in results]

    print("\n===== Load Test Results =====")
    print(f"Total requests : {len(results)}")
    print(f"Successful     : {successful}")
    print(f"Failed         : {failed}")
    print(f"Total time     : {total_time:.2f} seconds")
    print(f"Requests/sec   : {len(results) / total_time:.2f}")
    print(f"Min latency    : {min(latencies):.3f} seconds")
    print(f"Avg latency    : {sum(latencies) / len(latencies):.3f} seconds")
    print(f"Max latency    : {max(latencies):.3f} seconds")
    print("=============================\n")


if __name__ == "__main__":
    asyncio.run(main())