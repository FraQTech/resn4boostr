from .common import *
from .events import *

# ============================================================
# summaries
# ============================================================

def summarize_method(name: str, scores: np.ndarray, alarms: np.ndarray, threshold: np.ndarray, events: List[Event], summary: Dict[str, float]) -> MethodResult:
    top = max(events, key=lambda e: e.peak_score) if events else None
    return MethodResult(name=name, scores=scores, threshold=threshold, alarms=alarms, events=events, summary=summary, top_event=top, extra={})


def build_summary_table(results: List[MethodResult], main_name: str) -> pd.DataFrame:
    main = next((r for r in results if r.name == main_name), None)
    main_alarms = main.alarms if main is not None else None
    main_top = main.top_event if main is not None else None
    rows = []
    for r in results:
        rows.append({
            "method": r.name,
            "events": r.summary.get("events", np.nan),
            "events_per_hour": r.summary.get("events_per_hour", np.nan),
            "alarm_duty_cycle": r.summary.get("alarm_duty_cycle", np.nan),
            "mean_event_duration_s": r.summary.get("mean_event_duration_s", np.nan),
            "median_event_duration_s": r.summary.get("median_event_duration_s", np.nan),
            "threshold_method": r.summary.get("threshold_method", ""),
            "chosen_quantile": r.summary.get("chosen_quantile", np.nan),
            "overlap_jaccard_vs_main": alarms_jaccard(r.alarms, main_alarms) if main_alarms is not None else np.nan,
            "hits_main_top_event": (True if r.name == main_name else (event_overlaps_any(main_top, r.events) if main_top is not None else False)),
            **r.extra,
        })
    return pd.DataFrame(rows)

