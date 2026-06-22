"""
ZaminAI Multi-Agent Orchestrator
Supports TWO AI backends — use whichever key is available:

  Gemini  (google-generativeai REST API) — FREE tier, recommended default
  Claude  (Anthropic SDK)                — optional upgrade, better reasoning

Both run a ReAct (Reason → Act → Observe) loop with the same 12 satellite tools.
"""
import json, logging, os, requests, time
from datetime import datetime
from typing import Generator

log = logging.getLogger(__name__)

MAX_TOOL_CALLS = 8
MAX_ITERATIONS = 10

# Anthropic model names
MODEL_SMART = "claude-sonnet-4-6"
MODEL_FAST  = "claude-haiku-4-5-20251001"

# Gemini model names — try newest first, fall back to older
GEMINI_MODELS = [
    "gemini-2.0-flash",
    "gemini-2.0-flash-exp",
    "gemini-1.5-flash-latest",
    "gemini-1.5-flash",
    "gemini-1.5-flash-001",
]
GEMINI_KEY   = os.environ.get("GEMINI_API_KEY", "")
GEMINI_URL   = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"


# ── Convert Claude tool schema → Gemini functionDeclarations format ────────────
def _claude_tools_to_gemini(claude_tools: list) -> list:
    """Gemini REST API expects functionDeclarations with uppercase type strings."""
    TYPE_MAP = {"string":"STRING","integer":"INTEGER","number":"NUMBER",
                "boolean":"BOOLEAN","object":"OBJECT","array":"ARRAY"}

    def _prop(v):
        t = TYPE_MAP.get(v.get("type","string"), "STRING")
        p = {"type": t, "description": v.get("description","")}
        # Arrays: Gemini needs items schema; use STRING for coords to avoid complexity
        if t == "ARRAY":
            p["items"] = {"type": "STRING"}
        return p

    decls = []
    for t in claude_tools:
        schema = t.get("input_schema", {})
        props  = {k: _prop(v) for k, v in schema.get("properties", {}).items()}
        decls.append({
            "name":        t["name"],
            "description": t["description"],
            "parameters": {
                "type":       "OBJECT",
                "properties": props,
                "required":   schema.get("required", []),
            }
        })
    return [{"functionDeclarations": decls}]


# ── Gemini ReAct loop ─────────────────────────────────────────────────────────
LANG_INSTRUCTION = {
    "fa": "\n\nمهم: تمام پاسخ خود را به زبان دری بنویسید. هیچ انگلیسی استفاده نکنید.",
    "ps": "\n\nمهم: ټول ځواب دې پښتو ژبه ولیکئ. انګلیسي مه کاروئ.",
    "en": "",
}


def _inject_lang(system: str, language: str) -> str:
    """Append language instruction to system prompt."""
    return system + LANG_INSTRUCTION.get(language, "")


def run_gemini_agent(
    question:     str,
    system:       str,
    claude_tools: list,
    tool_context: dict,
    model:        str = None,
    language:     str = "en",
) -> dict:
    """
    ReAct loop using Gemini (free). Converts tool schemas automatically.
    Returns: {answer, tool_calls, iterations}
    """
    from agents.tools import execute_tool

    # Find a working Gemini model
    models_to_try = [model] if model else GEMINI_MODELS
    working_model = None
    for m in models_to_try:
        test_url = GEMINI_URL.format(model=m, key=GEMINI_KEY)
        try:
            probe = requests.post(test_url, json={
                "contents": [{"role":"user","parts":[{"text":"hi"}]}],
                "generationConfig": {"maxOutputTokens": 5}
            }, timeout=15)
            if probe.status_code == 200:
                working_model = m
                log.info(f"[Gemini] Using model: {m}")
                break
            log.warning(f"[Gemini] Model {m}: HTTP {probe.status_code}")
        except Exception as e:
            log.warning(f"[Gemini] Model {m}: {e}")

    if not working_model:
        return {"answer": "No working Gemini model found. Check your GEMINI_API_KEY.",
                "tool_calls": [], "iterations": 0}

    gemini_tools    = _claude_tools_to_gemini(claude_tools)
    tool_call_count = 0
    all_tool_calls  = []
    iterations      = 0

    system_with_lang = _inject_lang(system, language)
    contents = [
        {"role": "user", "parts": [{"text": f"[System]\n{system_with_lang}\n\n[Question]\n{question}"}]},
    ]

    url = GEMINI_URL.format(model=working_model, key=GEMINI_KEY)

    while iterations < MAX_ITERATIONS:
        iterations += 1

        payload = {
            "contents":           contents,
            "tools":              gemini_tools,
            "generationConfig":   {"maxOutputTokens": 2048, "temperature": 0.2},
        }

        try:
            resp = requests.post(url, json=payload, timeout=90)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            log.error(f"Gemini API error: {e}")
            return {"answer": f"Gemini API error: {e}", "tool_calls": all_tool_calls,
                    "iterations": iterations}

        candidates = data.get("candidates", [])
        if not candidates:
            break

        candidate = candidates[0]
        content   = candidate.get("content", {})
        parts     = content.get("parts", [])
        finish    = candidate.get("finishReason", "STOP")

        # Append model response to history
        contents.append({"role": "model", "parts": parts})

        # Check for function calls
        fn_calls = [p for p in parts if "functionCall" in p]
        text_parts= [p["text"] for p in parts if "text" in p]

        if not fn_calls:
            # Final answer
            answer = " ".join(text_parts).strip()
            return {"answer": answer, "tool_calls": all_tool_calls,
                    "iterations": iterations}

        # Execute each function call and collect results
        fn_responses = []
        for part in fn_calls:
            fc   = part["functionCall"]
            name = fc["name"]
            args = fc.get("args", {})

            if tool_call_count >= MAX_TOOL_CALLS:
                result = {"error": "Tool call limit reached"}
            else:
                tool_call_count += 1
                log.info(f"[Gemini] Tool {tool_call_count}: {name}({str(args)[:80]})")
                result_str = execute_tool(name, args, tool_context)
                result     = json.loads(result_str) if result_str.startswith("{") else {"result": result_str}
                all_tool_calls.append({"tool": name, "input": args, "output": result})

            fn_responses.append({
                "functionResponse": {
                    "name":     name,
                    "response": {"content": result},
                }
            })

        contents.append({"role": "user", "parts": fn_responses})

    return {"answer": "Agent reached max iterations.", "tool_calls": all_tool_calls,
            "iterations": iterations}


def run_gemini_streaming(
    question:     str,
    system:       str,
    claude_tools: list,
    tool_context: dict,
    model:        str = None,
    language:     str = "en",
) -> Generator[dict, None, None]:
    """Streaming wrapper for Gemini agent — yields SSE-compatible dicts."""
    from agents.tools import execute_tool

    yield {"type": "thinking", "text": "Querying Gemini agent…"}

    out = run_gemini_agent(question, system, claude_tools, tool_context,
                           model=model, language=language)

    for tc in out.get("tool_calls", []):
        yield {"type": "tool_call", "tool": tc["tool"], "input": tc["input"]}
        yield {"type": "tool_result","tool": tc["tool"], "data": tc["output"]}

    yield {"type": "answer",    "text": out.get("answer", "")}
    yield {"type": "done",      "iterations": out.get("iterations", 1),
           "tool_calls": len(out.get("tool_calls", []))}


def run_agent(
    question:    str,
    system:      str,
    tools:       list,
    tool_context:dict,
    anthropic_client,
    model:       str = MODEL_SMART,
    history:     list = None,
    max_tokens:  int  = 2048,
    language:    str  = "en",
) -> dict:
    """
    Run a single agentic turn: ReAct loop until Claude returns a final answer.
    Returns: {answer, tool_calls, usage, iterations}
    """
    messages         = list(history or []) + [{"role": "user", "content": question}]
    tool_call_count  = 0
    all_tool_calls   = []
    iterations       = 0
    system_with_lang = _inject_lang(system, language)

    from agents.tools import execute_tool

    while iterations < MAX_ITERATIONS:
        iterations += 1

        response = anthropic_client.messages.create(
            model      = model,
            max_tokens = max_tokens,
            system     = system_with_lang,
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
