"""Session-only, redacted trace records for the Streamlit RAG pipeline."""

import json
import re
from functools import lru_cache


STAGES = (
    ("input", "1. Хэрэглэгчийн оролт ба session history"),
    ("fast_check", "2. Local fast check"),
    ("router", "3. Semantic router"),
    ("policy", "4. Хүсэлт тус бүрийн restriction"),
    ("resolver", "5. Context resolver"),
    ("strategy", "6. Search strategy ба query set"),
    ("retrieval", "7. Vector Store retrieval"),
    ("answer", "8. Answer generation"),
    ("session_output", "9. Session-д хадгалсан үр дүн"),
)

_SECRET_KEY = re.compile(
    r"(api.?key|authorization|access.?token|secret|vector.?store.?id)", re.I
)


def create_pipeline_trace(prompt, history, *, secret_values=()):
    """Create a trace with the actual message and model-visible prior turns."""
    trace = {
        "schema_version": 1,
        "evidence_coverage": "Нотолгооны хүрэлцээ тусад нь үнэлэгдээгүй",
        "steps": [
            {
                "id": stage_id,
                "title": title,
                "status": "skipped",
                "input": None,
                "processing": "Энэ алхам хараахан ажиллаагүй.",
                "output": None,
                "duration_ms": None,
                "model": None,
                "usage": None,
                "handoff": None,
            }
            for stage_id, title in STAGES
        ],
        "_secret_values": [value for value in secret_values if isinstance(value, str) and value],
    }
    update_step(
        trace,
        "input",
        status="complete",
        input={"current_user_message": prompt, "session_history": history},
        processing="Streamlit session-ээс хэрэглэгчийн шинэ turn ба өмнөх history-г уншив.",
        output={"accepted_user_message": prompt},
        handoff={"current_question": prompt, "history_for_routing": history},
    )
    return trace


def update_step(
    trace,
    step_id,
    *,
    status,
    input=None,
    processing="",
    output=None,
    duration_ms=None,
    model=None,
    usage=None,
    handoff=None,
    error=None,
):
    """Replace one stage record while retaining the fixed nine-stage shape."""
    for step in trace["steps"]:
        if step["id"] == step_id:
            step.update(
                {
                    "status": status,
                    "input": input,
                    "processing": processing,
                    "output": output,
                    "duration_ms": duration_ms,
                    "model": model,
                    "usage": usage,
                    "handoff": handoff,
                    "error": error,
                }
            )
            return
    raise KeyError(f"Unknown pipeline stage: {step_id}")


def skip_steps(trace, step_ids, reason):
    for step_id in step_ids:
        update_step(trace, step_id, status="skipped", processing=reason)


def fail_step(trace, step_id, error, *, input=None, duration_ms=None, model=None):
    update_step(
        trace,
        step_id,
        status="error",
        input=input,
        processing="Алхам алдаагаар дууссан.",
        error={"type": type(error).__name__, "message": str(error)},
        duration_ms=duration_ms,
        model=model,
    )


def public_trace(trace):
    """Remove trace-only redaction state before keeping data in session state."""
    secret_values = trace.get("_secret_values", [])

    def clean(value, key=""):
        if _SECRET_KEY.search(key):
            return "[redacted]"
        if isinstance(value, dict):
            return {str(k): clean(v, str(k)) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(item) for item in value]
        if isinstance(value, str):
            for secret in secret_values:
                value = value.replace(secret, "[redacted]")
            return value
        if value is None or isinstance(value, (bool, int, float)):
            return value
        try:
            return json.loads(json.dumps(value, ensure_ascii=False, default=str))
        except (TypeError, ValueError):
            return str(value)

    return clean({key: value for key, value in trace.items() if key != "_secret_values"})


def render_pipeline_trace(trace, *, key):
    """Render only opened stages; toggling the UI never invokes the RAG runtime."""
    import streamlit as st

    outer = st.expander("Pipeline trace", expanded=False, on_change="rerun", key=key)
    if not outer.open:
        return
    with outer:
        _render_copy_button(trace, key)
        st.caption(
            f"Schema v{trace.get('schema_version', 1)} · "
            f"Нотолгооны хүрэлцээ: {trace.get('evidence_coverage', 'үнэлэгдээгүй')}"
        )
        statuses = {"complete": "Амжилттай", "skipped": "Алгассан", "error": "Алдаатай"}
        for step in trace.get("steps", []):
            status_label = statuses.get(step.get("status"), step.get("status", ""))
            with st.container(border=True):
                st.markdown(f"### {step.get('title', step.get('id'))} · {status_label}")
                if step.get("duration_ms") is not None:
                    st.caption(f"Хугацаа: {step['duration_ms']} ms")
                if step.get("model"):
                    st.caption(f"Model: {step['model']}")
                if step.get("usage"):
                    st.caption("Token usage: " + json.dumps(step["usage"], ensure_ascii=False))
                st.markdown("**Input**")
                _render_value(step.get("input"))
                st.markdown("**Processing / status**")
                st.write(step.get("processing", ""))
                if step.get("error"):
                    st.error(f"{step['error'].get('type')}: {step['error'].get('message')}")
                st.markdown("**Output**")
                _render_value(step.get("output"))
                st.markdown("**Дараагийн алхамд дамжсан утга**")
                _render_value(step.get("handoff"))


def _render_value(value):
    import streamlit as st

    if value is None:
        st.caption("Утга байхгүй")
    elif isinstance(value, str):
        st.code(value, language=None)
    else:
        st.code(json.dumps(value, ensure_ascii=False, indent=2), language="json")


def _render_copy_button(trace, key):
    """Add a browser clipboard button for the complete, already-open trace."""
    trace_json = json.dumps(trace, ensure_ascii=False, indent=2)
    _copy_trace_component()(
        data={"trace_json": trace_json},
        key=f"{key}-copy-all",
    )


@lru_cache(maxsize=1)
def _copy_trace_component():
    import streamlit as st

    return st.components.v2.component(
        "pipeline_trace_copy_button",
        html='<button type="button">Бүх trace-г хуулах</button>',
        css="""
            button {
                padding: 7px 12px;
                border: 1px solid #62666d;
                border-radius: 6px;
                background: #202329;
                color: #f4f4f5;
                cursor: pointer;
                font-size: 14px;
            }
        """,
        js="""
            export default function(component) {
                const {data, parentElement} = component;
                const button = parentElement.querySelector("button");
                const copy = async () => {
                    let copied = false;
                    try {
                        await navigator.clipboard.writeText(data.trace_json);
                        copied = true;
                    } catch (_) {
                        const field = document.createElement("textarea");
                        field.value = data.trace_json;
                        field.style.position = "fixed";
                        field.style.left = "-9999px";
                        document.body.appendChild(field);
                        field.select();
                        copied = document.execCommand("copy");
                        field.remove();
                    }
                    button.textContent = copied ? "Хуулагдлаа ✓" : "Хуулах боломжгүй";
                };
                button.addEventListener("click", copy);
                return () => button.removeEventListener("click", copy);
            }
        """,
    )
