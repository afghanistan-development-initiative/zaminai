"""
ZaminAI Multi-Agent Orchestrator
Runs a Claude ReAct (Reason → Act → Observe) loop with specialist tool access.
Supports: farmer advisory, officer analysis, field monitoring, autonomous alerts.
"""
import json, logging, time
from datetime import datetime
from typing import Generator

log = logging.getLogger(__name__)

MAX_TOOL_CALLS = 8          # cost guard per conversation turn
MAX_ITERATIONS = 10         # loop iteration limit
MODEL_SMART    = "claude-sonnet-4-6"   # main reasoning
MODEL_FAST     = "claude-haiku-4-5-20251001"   # tool execution / monitoring


def run_agent(
    question:    str,
    system:      str,
    tools:       list,
    tool_context:dict,
    anthropic_client,
    model:       str = MODEL_SMART,
    history:     list = None,
    max_tokens:  int  = 2048,
) -> dict:
    """
    Run a single agentic turn: ReAct loop until Claude returns a final answer.
    Returns: {answer, tool_calls, usage, iterations}
    """
    messages = list(history or []) + [{"role": "user", "content": question}]
    tool_call_count = 0
    all_tool_calls  = []
    iterations      = 0

    from agents.tools import execute_tool

    while iterations < MAX_ITERATIONS:
        iterations += 1

        response = anthropic_client.messages.create(
            model      = model,
            max_tokens = max_tokens,
            system     = system,
            tools      = tools,
            messages   = messages,
        )

        # Collect assistant message
        messages.append({"role": "assistant", "content": response.content})

        # ── Final answer ──────────────────────────────────────────────────────
        if response.stop_reason in ("end_turn", "stop_sequence"):
            answer = " ".join(
                b.text for b in response.content
                if hasattr(b, "text")
            ).strip()
            return {
                "answer":      answer,
                "tool_calls":  all_tool_calls,
                "iterations":  iterations,
                "usage":       {
                    "input_tokens":  response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                },
            }

        # ── Tool use block ────────────────────────────────────────────────────
        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue

                if tool_call_count >= MAX_TOOL_CALLS:
                    tool_results.append({
                        "type":        "tool_result",
                        "tool_use_id": block.id,
                        "content":     json.dumps({"error": "Tool call limit reached"}),
                    })
                    continue

                tool_call_count += 1
                log.info(f"[Agent] Tool call {tool_call_count}: {block.name}({json.dumps(block.input)[:120]})")

                result_str = execute_tool(block.name, block.input, tool_context)
                all_tool_calls.append({
                    "tool":   block.name,
                    "input":  block.input,
                    "output": json.loads(result_str) if result_str.startswith("{") else result_str,
                })

                tool_results.append({
                    "type":        "tool_result",
                    "tool_use_id": block.id,
                    "content":     result_str,
                })

            messages.append({"role": "user", "content": tool_results})
        else:
            # Unexpected stop reason
            break

    return {
        "answer":     "Agent reached maximum iterations without a final answer.",
        "tool_calls": all_tool_calls,
        "iterations": iterations,
    }


def run_streaming_agent(
    question:     str,
    system:       str,
    tools:        list,
    tool_context: dict,
    anthropic_client,
    model: str = MODEL_SMART,
    history: list = None,
) -> Generator[dict, None, None]:
    """
    Streaming version of run_agent. Yields server-sent event dicts:
    {type: 'thinking'|'tool_call'|'tool_result'|'answer'|'done', ...}
    """
    from agents.tools import execute_tool

    messages = list(history or []) + [{"role": "user", "content": question}]
    tool_call_count = 0
    iterations      = 0

    while iterations < MAX_ITERATIONS:
        iterations += 1
        response = anthropic_client.messages.create(
            model      = model,
            max_tokens = 2048,
            system     = system,
            tools      = tools,
            messages   = messages,
        )
        messages.append({"role": "assistant", "content": response.content})

        # Yield any text blocks as thinking/partial answer
        for block in response.content:
            if hasattr(block, "text") and block.text:
                yield {"type": "thinking", "text": block.text}

        if response.stop_reason in ("end_turn", "stop_sequence"):
            answer = " ".join(
                b.text for b in response.content if hasattr(b, "text")
            ).strip()
            yield {"type": "answer", "text": answer}
            yield {"type": "done", "iterations": iterations,
                   "tool_calls": tool_call_count}
            return

        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                tool_call_count += 1
                yield {"type": "tool_call", "tool": block.name,
                       "input": block.input, "n": tool_call_count}

                if tool_call_count > MAX_TOOL_CALLS:
                    result_str = json.dumps({"error": "Tool limit reached"})
                else:
                    result_str = execute_tool(block.name, block.input, tool_context)
                    result_obj = json.loads(result_str) if result_str.startswith("{") else {}
                    yield {"type": "tool_result", "tool": block.name, "data": result_obj}

                tool_results.append({
                    "type":        "tool_result",
                    "tool_use_id": block.id,
                    "content":     result_str,
                })
            messages.append({"role": "user", "content": tool_results})
        else:
            break

    yield {"type": "done", "iterations": iterations}


# ── Autonomous Field Monitor Loop ────────────────────────────────────────────
def run_field_monitor(tool_context: dict, anthropic_client) -> dict:
    """
    Autonomously monitor all registered farmer fields.
    Detects NDVI drops, generates natural-language alerts, saves to DB.
    Called by /agent/monitor endpoint (cron or manual trigger).
    """
    from agents.tools import execute_tool, TOOLS
    from agents.prompts import MONITOR_AGENT_PROMPT

    results   = {"checked": 0, "alerts": [], "errors": []}
    get_all_fields = tool_context.get("get_all_fields_fn")
    if not get_all_fields:
        return {"error": "Database not available"}

    try:
        all_fields = get_all_fields()
    except Exception as e:
        return {"error": f"Could not load fields: {e}"}

    for field in all_fields:
        try:
            coords    = field.get("coords", [])
            farmer_id = field.get("farmer_id")
            label     = field.get("label", "Field")
            if not coords or not farmer_id or len(coords) < 3:
                continue

            results["checked"] += 1

            # Build monitoring question
            question = (
                f"Analyse this field: '{label}' (farmer_id: {farmer_id}). "
                f"Coordinates: {coords[:4]}... "
                f"Current date: {datetime.now().strftime('%Y-%m-%d')}. "
                f"Check satellite health, compare to 14-day trend, assess risk level, "
                f"and if WARNING or CRITICAL: save an alert recommendation for the farmer."
            )

            out = run_agent(
                question     = question,
                system       = MONITOR_AGENT_PROMPT,
                tools        = TOOLS,
                tool_context = tool_context,
                anthropic_client = anthropic_client,
                model        = MODEL_FAST,
                max_tokens   = 512,
            )

            if out.get("answer"):
                results["alerts"].append({
                    "field":     label,
                    "farmer_id": farmer_id,
                    "summary":   out["answer"][:200],
                    "tool_calls": len(out.get("tool_calls", [])),
                })

            time.sleep(1)  # avoid GEE rate limits

        except Exception as e:
            results["errors"].append({"field": field.get("label"), "error": str(e)})

    return results


# ── Officer Regional Report Loop ─────────────────────────────────────────────
def run_weekly_officer_report(province: str, country: str,
                               tool_context: dict, anthropic_client) -> dict:
    """
    Generate a weekly satellite intelligence report for a province.
    Compares current week to previous 4 weeks, flags anomalies.
    """
    from agents.tools import TOOLS
    from agents.prompts import OFFICER_AGENT_PROMPT

    question = (
        f"Generate a weekly agricultural intelligence report for {province}, {country}. "
        f"Today is {datetime.now().strftime('%Y-%m-%d')}. "
        f"1. Query current satellite data for the region. "
        f"2. Get the NDVI trend for the past 2 years to identify if conditions are improving or degrading. "
        f"3. Get land cover breakdown to see what crops dominate. "
        f"4. Calculate a regional health score. "
        f"5. Identify the top 3 risks or issues visible from satellite data. "
        f"6. Give 3 actionable recommendations for field officers this week. "
        f"Format as a structured report with sections."
    )

    return run_agent(
        question     = question,
        system       = OFFICER_AGENT_PROMPT,
        tools        = TOOLS,
        tool_context = tool_context,
        anthropic_client = anthropic_client,
        model        = MODEL_SMART,
        max_tokens   = 2000,
    )
