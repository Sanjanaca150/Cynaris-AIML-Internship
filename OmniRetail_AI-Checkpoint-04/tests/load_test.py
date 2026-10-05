import time
from concurrent.futures import ThreadPoolExecutor

import httpx


URL = "http://127.0.0.1:8000/predict"

PAYLOAD = {
    "product_views": 5,
    "add_to_cart": 2,
    "session_duration_mins": 8.5,
    "return_visitor": 1,
    "discount_applied": 1,
    "recommendation_clicked": 1,
    "city": "Delhi",
    "device": "mobile",
    "category": "electronics",
}


def send_request():
    start = time.perf_counter()

    try:
        response = httpx.post(
            URL,
            json=PAYLOAD,
            timeout=30.0,
        )

        elapsed = time.perf_counter() - start

        return {
            "status_code": response.status_code,
            "latency": elapsed,
        }

    except Exception as error:
        return {
            "status_code": 0,
            "latency": None,
            "error": str(error),
        }


def run_load_test():
    total_requests = 50
    concurrent_requests = 50

    print("=" * 60)
    print("OmniRetail AI - Checkpoint 4 Load Test")
    print("=" * 60)
    print(f"Total requests      : {total_requests}")
    print(f"Concurrent requests : {concurrent_requests}")
    print()

    overall_start = time.perf_counter()

    with ThreadPoolExecutor(
        max_workers=concurrent_requests
    ) as executor:

        results = list(
            executor.map(
                lambda _: send_request(),
                range(total_requests),
            )
        )

    overall_time = time.perf_counter() - overall_start

    successful = [
        result
        for result in results
        if result["status_code"] == 200
    ]

    failed = [
        result
        for result in results
        if result["status_code"] != 200
    ]

    latencies = [
        result["latency"]
        for result in successful
        if result["latency"] is not None
    ]

    if latencies:
        average_latency = sum(latencies) / len(latencies)
        min_latency = min(latencies)
        max_latency = max(latencies)
    else:
        average_latency = 0
        min_latency = 0
        max_latency = 0

    throughput = (
        len(successful) / overall_time
        if overall_time > 0
        else 0
    )

    print(f"Successful requests : {len(successful)}")
    print(f"Failed requests     : {len(failed)}")
    print(f"Total test time     : {overall_time:.4f} seconds")
    print(f"Average latency     : {average_latency * 1000:.2f} ms")
    print(f"Minimum latency     : {min_latency * 1000:.2f} ms")
    print(f"Maximum latency     : {max_latency * 1000:.2f} ms")
    print(f"Throughput          : {throughput:.2f} requests/sec")
    print()

    if failed:
        print("Failures:")
        for result in failed:
            print(result)

    assert len(successful) == total_requests
    assert len(failed) == 0

    print("50-concurrent-request load test: PASSED")


if __name__ == "__main__":
    run_load_test()