from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd


@dataclass
class QAQCSummary:
    duplicate_intervals: int
    gaps: int
    overlaps: int
    non_positive_lengths: int
    orientation_inconsistencies: int
    missing_columns: List[str]

    def to_dict(self) -> Dict[str, object]:
        return {
            "duplicate_intervals": self.duplicate_intervals,
            "gaps": self.gaps,
            "overlaps": self.overlaps,
            "non_positive_lengths": self.non_positive_lengths,
            "orientation_inconsistencies": self.orientation_inconsistencies,
            "missing_columns": ", ".join(self.missing_columns) if self.missing_columns else "",
        }


def _missing_columns(df: pd.DataFrame, cols: Iterable[str]) -> List[str]:
    return [col for col in cols if col not in df.columns]


def detect_duplicate_intervals(
    df: pd.DataFrame,
    collar_col: str,
    from_col: str,
    to_col: str,
) -> pd.DataFrame:
    missing = _missing_columns(df, [collar_col, from_col, to_col])
    if missing:
        return pd.DataFrame()

    subset = [collar_col, from_col, to_col]
    dup_mask = df.duplicated(subset=subset, keep=False)
    if not dup_mask.any():
        return pd.DataFrame()

    out = df.loc[dup_mask, subset].copy()
    out["issue_type"] = "duplicate_interval"
    return out


def find_gaps_overlaps(
    df: pd.DataFrame,
    collar_col: str,
    from_col: str,
    to_col: str,
    tolerance: float = 0.0,
) -> pd.DataFrame:
    missing = _missing_columns(df, [collar_col, from_col, to_col])
    if missing:
        return pd.DataFrame()

    work = df[[collar_col, from_col, to_col]].copy()
    work[from_col] = pd.to_numeric(work[from_col], errors="coerce")
    work[to_col] = pd.to_numeric(work[to_col], errors="coerce")
    work = work.dropna(subset=[collar_col, from_col, to_col])
    if work.empty:
        return pd.DataFrame()

    records: List[Dict[str, object]] = []
    for collar, grp in work.groupby(collar_col):
        grp_sorted = grp.sort_values(by=[from_col, to_col]).reset_index(drop=True)
        for idx in range(len(grp_sorted) - 1):
            current = grp_sorted.iloc[idx]
            nxt = grp_sorted.iloc[idx + 1]
            delta = float(nxt[from_col] - current[to_col])
            if abs(delta) <= tolerance:
                continue
            issue_type = "gap" if delta > 0 else "overlap"
            records.append(
                {
                    "issue_type": issue_type,
                    collar_col: collar,
                    "from": float(current[from_col]),
                    "to": float(current[to_col]),
                    "next_from": float(nxt[from_col]),
                    "next_to": float(nxt[to_col]),
                    "delta": delta,
                }
            )
    return pd.DataFrame.from_records(records)


def find_non_positive_lengths(
    df: pd.DataFrame,
    collar_col: str,
    from_col: str,
    to_col: str,
) -> pd.DataFrame:
    missing = _missing_columns(df, [collar_col, from_col, to_col])
    if missing:
        return pd.DataFrame()

    work = df[[collar_col, from_col, to_col]].copy()
    work[from_col] = pd.to_numeric(work[from_col], errors="coerce")
    work[to_col] = pd.to_numeric(work[to_col], errors="coerce")
    work = work.dropna(subset=[from_col, to_col])
    if work.empty:
        return pd.DataFrame()

    work["length"] = work[to_col] - work[from_col]
    mask = work["length"] <= 0
    if not mask.any():
        return pd.DataFrame()

    out = work.loc[mask, [collar_col, from_col, to_col, "length"]].copy()
    out["issue_type"] = "non_positive_length"
    return out


def find_orientation_inconsistencies(
    df: pd.DataFrame,
    collar_col: str,
    azimuth_col: Optional[str] = None,
    dip_col: Optional[str] = None,
) -> pd.DataFrame:
    cols = [collar_col]
    if azimuth_col:
        cols.append(azimuth_col)
    if dip_col:
        cols.append(dip_col)

    missing = _missing_columns(df, cols)
    if missing:
        return pd.DataFrame()

    work = df[cols].copy()
    for col in [azimuth_col, dip_col]:
        if col:
            work[col] = pd.to_numeric(work[col], errors="coerce")

    records: List[Dict[str, object]] = []
    for collar, grp in work.groupby(collar_col):
        az_values = (
            sorted(grp[azimuth_col].dropna().unique().tolist()) if azimuth_col else []
        )
        dip_values = sorted(grp[dip_col].dropna().unique().tolist()) if dip_col else []
        az_count = len(az_values)
        dip_count = len(dip_values)
        if (azimuth_col and az_count > 1) or (dip_col and dip_count > 1):
            records.append(
                {
                    "issue_type": "orientation_inconsistency",
                    collar_col: collar,
                    "azimuth_values": ", ".join(map(str, az_values)) if azimuth_col else "",
                    "dip_values": ", ".join(map(str, dip_values)) if dip_col else "",
                    "azimuth_count": az_count,
                    "dip_count": dip_count,
                }
            )
    return pd.DataFrame.from_records(records)


def build_qaqc_report(
    df: pd.DataFrame,
    collar_col: str,
    from_col: str,
    to_col: str,
    azimuth_col: Optional[str] = None,
    dip_col: Optional[str] = None,
    tolerance: float = 0.0,
) -> Tuple[pd.DataFrame, QAQCSummary]:
    required_cols = [collar_col, from_col, to_col]
    missing = _missing_columns(df, required_cols)
    if azimuth_col:
        missing.extend([col for col in _missing_columns(df, [azimuth_col])])
    if dip_col:
        missing.extend([col for col in _missing_columns(df, [dip_col])])
    missing = sorted(set(missing))

    duplicates = detect_duplicate_intervals(df, collar_col, from_col, to_col)
    gaps_overlaps = find_gaps_overlaps(df, collar_col, from_col, to_col, tolerance=tolerance)
    non_positive = find_non_positive_lengths(df, collar_col, from_col, to_col)
    orientation = find_orientation_inconsistencies(df, collar_col, azimuth_col, dip_col)

    report_frames = [duplicates, gaps_overlaps, non_positive, orientation]
    report = pd.concat([frame for frame in report_frames if not frame.empty], ignore_index=True)

    summary = QAQCSummary(
        duplicate_intervals=int(len(duplicates)),
        gaps=int((gaps_overlaps["issue_type"] == "gap").sum()) if not gaps_overlaps.empty else 0,
        overlaps=int((gaps_overlaps["issue_type"] == "overlap").sum()) if not gaps_overlaps.empty else 0,
        non_positive_lengths=int(len(non_positive)),
        orientation_inconsistencies=int(len(orientation)),
        missing_columns=missing,
    )
    return report, summary


def export_qaqc_report(
    report: pd.DataFrame,
    summary: QAQCSummary,
    output_dir: str,
    report_name: str = "qaqc_report.csv",
    summary_name: str = "qaqc_summary.csv",
) -> Tuple[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, report_name)
    summary_path = os.path.join(output_dir, summary_name)

    if report.empty:
        pd.DataFrame(columns=["issue_type"]).to_csv(report_path, index=False)
    else:
        report.to_csv(report_path, index=False)

    summary_df = pd.DataFrame(
        [{"metric": key, "value": value} for key, value in summary.to_dict().items()]
    )
    summary_df.to_csv(summary_path, index=False)
    return report_path, summary_path


def update_manifest(manifest_path: str, metrics: Dict[str, object]) -> None:
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    payload: Dict[str, object] = {}
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as handle:
            try:
                payload = json.load(handle)
            except json.JSONDecodeError:
                payload = {}
    payload["qaqc"] = {
        "updated_at": datetime.utcnow().isoformat() + "Z",
        "metrics": metrics,
    }
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def run_qaqc(
    df: pd.DataFrame,
    collar_col: str,
    from_col: str,
    to_col: str,
    azimuth_col: Optional[str] = None,
    dip_col: Optional[str] = None,
    tolerance: float = 0.0,
    output_dir: str = "outputs/qaqc",
    manifest_path: str = "outputs/manifest.json",
) -> Dict[str, object]:
    report, summary = build_qaqc_report(
        df,
        collar_col=collar_col,
        from_col=from_col,
        to_col=to_col,
        azimuth_col=azimuth_col,
        dip_col=dip_col,
        tolerance=tolerance,
    )
    report_path, summary_path = export_qaqc_report(report, summary, output_dir)

    metrics = summary.to_dict()
    metrics.update({"report_path": report_path, "summary_path": summary_path})
    update_manifest(manifest_path, metrics)
    return metrics
