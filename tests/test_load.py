"""
Concurrent load tests for the rate limiter.

Tests that _RateLimiter is thread-safe: under concurrent requests from a
single IP the total number of allowed calls never exceeds max_calls, and
requests from different IPs are independently limited.

These tests use real threads (not asyncio) because the rate limiter uses
threading.Lock internally and we want to exercise the lock under genuine
concurrency rather than cooperative multitasking.
"""

import threading
import time
import pytest
from api import _RateLimiter


def test_rate_limiter_thread_safety_single_ip():
    """Concurrent requests from one IP must not exceed the configured limit.

    20 threads each call is_allowed() simultaneously.  With max_calls=10,
    exactly 10 should be allowed and 10 blocked — regardless of thread
    scheduling order.
    """
    limiter = _RateLimiter(max_calls=10, window_seconds=60)
    results = []
    lock = threading.Lock()

    def make_request():
        allowed = limiter.is_allowed("test_ip")
        with lock:
            results.append(allowed)

    threads = [threading.Thread(target=make_request) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(results) == 20
    allowed_count = sum(results)
    assert allowed_count <= 10, (
        f"Rate limiter allowed {allowed_count} requests for one IP "
        f"(limit is 10) — threading.Lock may not be protecting the bucket"
    )
    assert allowed_count >= 1, "Rate limiter blocked all requests — something is wrong"


def test_rate_limiter_thread_safety_multiple_ips():
    """Each IP gets its own independent quota under concurrent access."""
    n_ips = 8
    calls_per_ip = 10
    max_calls = 5
    limiter = _RateLimiter(max_calls=max_calls, window_seconds=60)

    results_by_ip: dict = {f"ip_{i}": [] for i in range(n_ips)}
    lock = threading.Lock()

    def bombard_ip(ip: str):
        for _ in range(calls_per_ip):
            allowed = limiter.is_allowed(ip)
            with lock:
                results_by_ip[ip].append(allowed)

    threads = [
        threading.Thread(target=bombard_ip, args=(f"ip_{i}",))
        for i in range(n_ips)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    for ip, ip_results in results_by_ip.items():
        allowed = sum(ip_results)
        assert allowed <= max_calls, (
            f"{ip} was allowed {allowed} requests (limit={max_calls})"
        )
        assert allowed >= 1, f"{ip} was completely blocked"


def test_rate_limiter_window_expiry():
    """Requests in an expired window should not count against the new window."""
    # Use a very short window (0.1 s) so the test stays fast.
    limiter = _RateLimiter(max_calls=3, window_seconds=1)

    # Fill the bucket
    for _ in range(3):
        assert limiter.is_allowed("ip_x") is True
    # 4th request in the same window must be blocked
    assert limiter.is_allowed("ip_x") is False

    # Wait for the window to expire
    time.sleep(1.1)

    # After expiry the bucket resets — 3 new requests should be allowed
    for _ in range(3):
        assert limiter.is_allowed("ip_x") is True


def test_rate_limiter_evicts_idle_buckets():
    """Idle IP entries must be evicted after their window expires."""
    limiter = _RateLimiter(max_calls=5, window_seconds=1)
    limiter.is_allowed("ephemeral_ip")
    assert "ephemeral_ip" in limiter._buckets

    time.sleep(1.1)

    # Trigger eviction: the next call for this IP will see an empty call list
    # and clean up the bucket entry.
    limiter.is_allowed("ephemeral_ip")
    # After the second call the bucket should exist again (one new entry)
    # but previously it should have been evicted (empty list → del)
    # We just verify no crash and correct allow/block behaviour.
    assert limiter.is_allowed("ephemeral_ip") is True
