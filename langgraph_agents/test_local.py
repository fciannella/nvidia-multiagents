"""Phase 1 local test: run all three agents in-process.

Simulates the full flow:
1. User sends a message
2. Router classifies intent (parallel in production, sequential here)
3. If tool needed: worker runs, writes progress to Store
4. Chitchat responds (aware of job state via Store)
5. After worker completes: chitchat delivers the result

Run from the langgraph_agents/ directory:
    cd langgraph_agents
    source ../.venv/bin/activate && python test_local.py
"""

import sys
import os
import time
from threading import Thread

sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv()

from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from loguru import logger

store = InMemoryStore()
checkpointer = MemorySaver()

SESSION_ID = "test-session"
CHAT_THREAD = "chat-001"
WORKER_THREAD = "worker-001"


def _inject(graph, *, cp=None, st=None):
    """Inject checkpointer/store into an @entrypoint Pregel for local testing.

    On the Platform these are provided by the server runtime; locally we
    set them directly on the Pregel instance.
    """
    if cp is not None:
        graph.checkpointer = cp
    if st is not None:
        graph.store = st
    return graph


def separator(title: str):
    logger.info(f"\n{'='*60}\n  {title}\n{'='*60}")


def test_router():
    """Test 1: Router classifies intent."""
    separator("TEST 1: Router - classify intent")

    from agents.router import graph as router_graph

    result = router_graph.invoke({
        "message": "Can you analyze my Q3 sales data for anomalies?",
        "recent_history": [
            {"role": "user", "content": "Hi there"},
            {"role": "assistant", "content": "Hello! How can I help?"},
        ],
    })
    logger.info(f"Router result: {result}")
    assert result.get("action") == "launch_tool", f"Expected launch_tool, got: {result}"
    logger.success("Router correctly identified tool call")
    return result


def test_router_no_tool():
    """Test 2: Router says no tool needed for casual chat."""
    separator("TEST 2: Router - no tool needed")

    from agents.router import graph as router_graph

    result = router_graph.invoke({
        "message": "Tell me a joke about programmers",
        "recent_history": [],
    })
    logger.info(f"Router result: {result}")
    assert result.get("action") == "none", f"Expected none, got: {result}"
    logger.success("Router correctly identified no tool needed")
    return result


def test_chitchat_basic():
    """Test 3: Chitchat responds conversationally."""
    separator("TEST 3: Chitchat - basic conversation")

    from agents.chitchat import graph as chitchat_graph
    _inject(chitchat_graph, cp=checkpointer, st=store)

    config = {"configurable": {"thread_id": CHAT_THREAD}}

    response = chitchat_graph.invoke({
        "session_id": SESSION_ID,
        "type": "user_message",
        "message": "Hey! How are you doing today?",
    }, config)

    logger.info(f"Chitchat response: {response}")
    assert isinstance(response, str) and len(response) > 0
    logger.success("Chitchat responded successfully")
    return response


def test_worker():
    """Test 4: Worker executes a tool and writes progress to Store."""
    separator("TEST 4: Worker - execute quick_lookup tool")

    from agents.worker import graph as worker_graph
    _inject(worker_graph, st=store)

    config = {"configurable": {"thread_id": WORKER_THREAD}}

    result = worker_graph.invoke({
        "tool": "quick_lookup",
        "params": {"query": "Q3 revenue spikes"},
        "session_id": SESSION_ID,
        "job_id": "job-test-001",
    }, config)

    logger.info(f"Worker result: {result}")
    assert result["status"] == "complete"

    items = store.search(("jobs", SESSION_ID))
    logger.info(f"Store items after worker: {len(items)}")
    for item in items:
        logger.info(f"  {item.key}: status={item.value.get('status')}")

    logger.success("Worker executed tool and wrote results to Store")
    return result


def test_chitchat_delivers_result():
    """Test 5: Chitchat delivers job result after worker completes."""
    separator("TEST 5: Chitchat - deliver job result")

    from agents.chitchat import graph as chitchat_graph
    _inject(chitchat_graph, cp=checkpointer, st=store)

    config = {"configurable": {"thread_id": CHAT_THREAD}}

    response = chitchat_graph.invoke({
        "session_id": SESSION_ID,
        "type": "job_result",
        "job_id": "job-test-001",
        "original_question": "Can you look up Q3 revenue spikes?",
    }, config)

    logger.info(f"Chitchat result delivery: {response}")
    assert isinstance(response, str) and len(response) > 0
    logger.success("Chitchat delivered job results")
    return response


def test_chitchat_followup():
    """Test 6: Chitchat handles a follow-up question about the result."""
    separator("TEST 6: Chitchat - follow-up question")

    from agents.chitchat import graph as chitchat_graph
    _inject(chitchat_graph, cp=checkpointer, st=store)

    config = {"configurable": {"thread_id": CHAT_THREAD}}

    response = chitchat_graph.invoke({
        "session_id": SESSION_ID,
        "type": "user_message",
        "message": "What was the confidence level of the top match?",
    }, config)

    logger.info(f"Follow-up response: {response}")
    assert isinstance(response, str) and len(response) > 0
    logger.success("Chitchat handled follow-up using Store context")
    return response


def test_full_flow():
    """Test 7: Simulates the full parallel flow (sequential here)."""
    separator("TEST 7: Full flow simulation")

    from agents.router import graph as router_graph
    from agents.worker import graph as worker_graph
    from agents.chitchat import graph as chitchat_graph

    fresh_store = InMemoryStore()
    fresh_checkpointer = MemorySaver()
    sid = "full-flow-session"

    _inject(chitchat_graph, cp=fresh_checkpointer, st=fresh_store)
    _inject(worker_graph, st=fresh_store)
    chat_config = {"configurable": {"thread_id": "chat-full"}}

    user_msg = "Please analyze my Q3 financial data, focusing on cost anomalies"

    logger.info(f"User: {user_msg}")

    # Step 1: Chitchat responds immediately
    chat_response = chitchat_graph.invoke({
        "session_id": sid,
        "type": "user_message",
        "message": user_msg,
    }, chat_config)
    logger.info(f"Chitchat (immediate): {chat_response}")

    # Step 2: Router classifies (in production this is parallel with step 1)
    router_result = router_graph.invoke({
        "message": user_msg,
        "recent_history": [],
    })
    logger.info(f"Router decision: {router_result}")

    if router_result.get("action") == "launch_tool":
        tool = router_result["tool"]
        params = router_result.get("params", {})
        job_id = "job-full-001"

        # Step 3: Worker runs (in production this is a background run)
        logger.info(f"Launching worker: tool={tool} job_id={job_id}")
        worker_result = worker_graph.invoke({
            "tool": tool,
            "params": params,
            "session_id": sid,
            "job_id": job_id,
            "context_summary": router_result.get("context_summary", ""),
        }, {"configurable": {"thread_id": "worker-full"}})
        logger.info(f"Worker completed: {worker_result['status']}")

        # Step 4: Deliver result via chitchat (triggered by webhook in production)
        delivery = chitchat_graph.invoke({
            "session_id": sid,
            "type": "job_result",
            "job_id": job_id,
            "original_question": user_msg,
        }, chat_config)
        logger.info(f"Chitchat (result delivery): {delivery}")

        # Step 5: Follow-up question
        followup = chitchat_graph.invoke({
            "session_id": sid,
            "type": "user_message",
            "message": "What was the biggest anomaly?",
        }, chat_config)
        logger.info(f"Chitchat (follow-up): {followup}")

    logger.success("Full flow completed successfully!")


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    tests = [
        test_router,
        test_router_no_tool,
        test_chitchat_basic,
        test_worker,
        test_chitchat_delivers_result,
        test_chitchat_followup,
        test_full_flow,
    ]

    passed = 0
    failed = 0
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            logger.error(f"FAILED: {test_fn.__name__}: {e}")
            failed += 1

    separator("RESULTS")
    logger.info(f"Passed: {passed}/{len(tests)}, Failed: {failed}/{len(tests)}")
    if failed:
        sys.exit(1)
