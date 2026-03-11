"""Workflow tools for the ReAct agent.

These are LangChain @tool functions that the workflow agent calls via
bind_tools(). They include business-logic tools (lookups, analysis) and
the special `ask_user` tool that triggers interrupt() for human-in-the-loop.

Tool functions use asyncio.sleep to simulate work. Long-running tools
write progress to the LangGraph Store so the chitchat agent can report
status while the tool is executing.
"""

import asyncio
import contextvars
import json
import uuid

from langchain_core.tools import tool
from langgraph.config import get_store
from langgraph.types import interrupt
from loguru import logger

workflow_session_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "workflow_session_id", default="default"
)


async def _write_progress(job_id: str, data: dict):
    session_id = workflow_session_id.get()
    store = get_store()
    await store.aput(("jobs", session_id), job_id, data)
    logger.info(
        f"[{job_id}] progress: {data.get('status')} "
        f"step={data.get('step')}/{data.get('total')} "
        f"session={session_id}"
    )


@tool
def ask_user(question: str) -> str:
    """Ask the user a follow-up question. Use this when you need information
    from the user to proceed, such as their name, date of birth, account
    details, a date range, or confirmation before taking an action.

    The question should be clear and specific. The user's answer will be
    returned as a string.
    """
    return interrupt({"question": question})


@tool
async def quick_lookup(query: str) -> str:
    """Search the database for records matching a query. Use for quick
    data retrieval: finding customer records, checking account info,
    looking up transactions, or verifying details.

    Returns a JSON string with results.
    """
    job_id = f"wf-{uuid.uuid4().hex[:6]}"
    total = 5

    for step in range(1, total + 1):
        await asyncio.sleep(1)

    result = {
        "records_found": 142,
        "top_match": f"Record matching '{query}'",
        "confidence": 0.94,
        "sample_records": [
            {"id": "rec-001", "name": query, "status": "active"},
            {"id": "rec-002", "name": f"{query} (secondary)", "status": "pending"},
        ],
    }
    logger.info(f"[{job_id}] quick_lookup complete: {query}")
    return json.dumps(result)


@tool
async def long_analysis(dataset: str, focus: str) -> str:
    """Run a detailed data analysis on a dataset with a specific focus area.
    Use for comprehensive reports, anomaly detection, trend analysis,
    quarterly reviews, or financial investigations.

    Takes 20-30 seconds. Returns a JSON string with findings.
    """
    job_id = f"wf-{uuid.uuid4().hex[:6]}"
    total = 10

    phase_names = [
        "Loading dataset",
        "Validating schema",
        "Computing statistics",
        "Detecting anomalies",
        "Running regression",
        "Cross-validating",
        "Generating clusters",
        "Ranking features",
        "Building summary",
        "Finalizing report",
    ]

    for step in range(1, total + 1):
        await asyncio.sleep(3)
        try:
            await _write_progress(job_id, {
                "status": "running",
                "tool": "long_analysis",
                "step": step,
                "total": total,
                "message": f"{phase_names[step - 1]}...",
            })
        except Exception:
            pass

    result = {
        "dataset": dataset,
        "focus": focus,
        "anomalies_found": 3,
        "top_anomaly": {"column": focus, "month": "August", "deviation": "+40%"},
        "secondary_anomalies": [
            {"column": focus, "month": "July", "deviation": "+12%"},
            {"column": focus, "month": "September", "deviation": "+8%"},
        ],
        "overall_trend": "upward",
        "summary": (
            f"Analysis of '{dataset}' focusing on '{focus}' found 3 anomalies. "
            f"The biggest was a 40% spike in August. July and September also "
            f"showed smaller deviations. Overall trend is upward."
        ),
    }

    try:
        await _write_progress(job_id, {
            "status": "complete",
            "tool": "long_analysis",
            "step": total,
            "total": total,
            "message": "Analysis complete.",
            "result": result,
        })
    except Exception:
        pass

    logger.info(f"[{job_id}] long_analysis complete: {dataset}/{focus}")
    return json.dumps(result)


@tool
async def train_model(model_name: str, dataset: str) -> str:
    """Train a machine learning model on a given dataset. Use when the
    user explicitly asks to train, fine-tune, or build a model.

    Takes about 60 seconds. Returns a JSON string with training results.
    """
    job_id = f"wf-{uuid.uuid4().hex[:6]}"
    total = 12

    for step in range(1, total + 1):
        epoch_loss = round(2.5 - (step * 0.18), 3)
        await asyncio.sleep(5)
        try:
            await _write_progress(job_id, {
                "status": "running",
                "tool": "train_model",
                "step": step,
                "total": total,
                "message": f"Training epoch {step}/{total}, loss={epoch_loss}",
            })
        except Exception:
            pass

    result = {
        "model_name": model_name,
        "dataset": dataset,
        "final_loss": 0.34,
        "accuracy": 0.91,
        "epochs_completed": total,
        "training_time_secs": 60,
        "summary": (
            f"Model '{model_name}' trained on '{dataset}' for {total} epochs. "
            f"Final loss: 0.34, accuracy: 91%. Ready for deployment."
        ),
    }

    try:
        await _write_progress(job_id, {
            "status": "complete",
            "tool": "train_model",
            "step": total,
            "total": total,
            "message": "Training complete.",
            "result": result,
        })
    except Exception:
        pass

    logger.info(f"[{job_id}] train_model complete: {model_name}")
    return json.dumps(result)


ALL_TOOLS = [ask_user, quick_lookup, long_analysis, train_model]
TOOL_NAMES = [t.name for t in ALL_TOOLS]
TOOLS_BY_NAME = {t.name: t for t in ALL_TOOLS}
