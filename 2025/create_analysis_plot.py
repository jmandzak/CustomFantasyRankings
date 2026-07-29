from argparse import ArgumentParser, Namespace
from pathlib import Path
import re

import pandas as pd
import plotly.graph_objects as go

POSITIONS = ("QB", "RB", "WR", "TE")
POSITION_COLORS = {"QB": "#2563eb", "RB": "#dc2626", "WR": "#059669", "TE": "#d97706"}


def parse_args() -> Namespace:
    parser = ArgumentParser(
        description="Create an interactive fantasy ranking analysis"
    )
    parser.add_argument("--predictions", required=True, help="Prediction CSV path")
    parser.add_argument("--actual", required=True, help="Actual-rank CSV path")
    parser.add_argument("--adp", required=True, help="ADP CSV path")
    parser.add_argument(
        "--ecr", default="ecr_predictions.csv", help="ECR prediction CSV path"
    )
    parser.add_argument("--output", default="analysis.html", help="Output HTML path")
    parser.add_argument(
        "--min-games",
        type=int,
        default=0,
        help="Exclude players with fewer than this many games played (default: 0)",
    )
    parser.add_argument(
        "--metric",
        choices=("total", "avg"),
        default="total",
        help="Initial actual-performance metric for charts (default: total)",
    )
    return parser.parse_args()


def normalize_name(value: object) -> str:
    text = str(value)
    if text.lower() in ("nan", "<na>", "none"):
        return ""
    text = text.upper().strip()
    text = re.sub(r"[^A-Z0-9 ]", "", text)
    return re.sub(r"\s+", " ", text)


def load_predictions(path: str) -> pd.DataFrame:
    source = pd.read_csv(path)
    required = {"POS RK", *POSITIONS}
    missing = required.difference(source.columns)
    if missing:
        raise ValueError(f"Prediction file is missing columns: {sorted(missing)}")

    frames = []
    for position in POSITIONS:
        frame = source[["POS RK", position]].rename(
            columns={"POS RK": "predicted_rank", position: "player"}
        )
        frame["position"] = position
        frames.append(frame)

    predictions = pd.concat(frames, ignore_index=True)
    predictions["player"] = predictions["player"].fillna("").astype(str).str.strip()
    predictions = predictions[predictions["player"].ne("")].copy()
    predictions["predicted_rank"] = pd.to_numeric(
        predictions["predicted_rank"], errors="coerce"
    )
    predictions = predictions.dropna(subset=["predicted_rank"])
    predictions["name_key"] = predictions["player"].map(normalize_name)
    return predictions


def load_actuals(path: str) -> pd.DataFrame:
    actuals = pd.read_csv(path)
    required = {"PLAYER", "GP", "AVG", "TTL"}
    missing = required.difference(actuals.columns)
    if missing:
        raise ValueError(f"Actual file is missing columns: {sorted(missing)}")

    actuals = actuals.rename(columns={"PLAYER": "actual_player"}).copy()
    actuals["name_key"] = actuals["actual_player"].map(normalize_name)
    for column in ("GP", "AVG", "TTL"):
        actuals[column] = pd.to_numeric(actuals[column], errors="coerce")
    return actuals


def load_adp(path: str) -> pd.DataFrame:
    adp = pd.read_csv(path)
    required = {"NAME", "ADP"}
    missing = required.difference(adp.columns)
    if missing:
        raise ValueError(f"ADP file is missing columns: {sorted(missing)}")

    adp = adp.rename(columns={"NAME": "adp_player"}).copy()
    adp["name_key"] = adp["adp_player"].map(normalize_name)
    adp["ADP"] = pd.to_numeric(adp["ADP"], errors="coerce")
    return adp.dropna(subset=["ADP"])


def load_ecr(path: str) -> pd.DataFrame:
    ecr = pd.read_csv(path)
    required = {"PLAYER NAME", "PPR_RK"}
    missing = required.difference(ecr.columns)
    if missing:
        raise ValueError(f"ECR file is missing columns: {sorted(missing)}")

    ecr = ecr.rename(columns={"PLAYER NAME": "ecr_player"}).copy()
    ecr["name_key"] = ecr["ecr_player"].map(normalize_name)
    ecr["PPR_RK"] = pd.to_numeric(ecr["PPR_RK"], errors="coerce")
    return ecr.dropna(subset=["PPR_RK"])


def calculate_analysis(
    predictions: pd.DataFrame,
    actuals: pd.DataFrame,
    adp: pd.DataFrame,
    ecr: pd.DataFrame,
    min_games: int,
) -> tuple[
    pd.DataFrame, list[str], list[str], list[str], list[str], list[str], list[str]
]:
    duplicate_names = (
        actuals.loc[actuals["name_key"].duplicated(keep=False), "name_key"]
        .drop_duplicates()
        .sort_values()
        .tolist()
    )
    actuals = actuals.drop_duplicates("name_key", keep=False).copy()
    duplicate_adp_names = (
        adp.loc[adp["name_key"].duplicated(keep=False), "name_key"]
        .drop_duplicates()
        .sort_values()
        .tolist()
    )
    adp = adp.drop_duplicates("name_key", keep=False).copy()
    duplicate_ecr_names = (
        ecr.loc[ecr["name_key"].duplicated(keep=False), "name_key"]
        .drop_duplicates()
        .sort_values()
        .tolist()
    )
    ecr = ecr.drop_duplicates("name_key", keep=False).copy()
    unmatched_predictions = sorted(
        set(predictions["name_key"]) - set(actuals["name_key"])
    )
    unmatched_adp = sorted(set(predictions["name_key"]) - set(adp["name_key"]))
    unmatched_ecr = sorted(set(predictions["name_key"]) - set(ecr["name_key"]))
    prediction_adp = predictions.merge(
        adp[["name_key", "ADP"]], on="name_key", how="left"
    )
    prediction_adp["adp_rank"] = prediction_adp.groupby("position")["ADP"].rank(
        method="min", ascending=True
    )
    prediction_sources = prediction_adp.merge(
        ecr[["name_key", "PPR_RK"]], on="name_key", how="left"
    )
    prediction_sources["ecr_rank"] = prediction_sources.groupby("position")[
        "PPR_RK"
    ].rank(method="min", ascending=True)
    matched = prediction_sources.merge(actuals, on="name_key", how="inner")
    matched = matched[matched["GP"].ge(min_games)].copy()

    if "POS" in actuals.columns:
        actuals["position"] = actuals["POS"].replace({"DST": "DEF"})
        actuals["actual_rank_total"] = actuals.groupby("position")["TTL"].rank(
            method="min", ascending=False
        )
        actuals["actual_rank_avg"] = actuals.groupby("position")["AVG"].rank(
            method="min", ascending=False
        )
        matched = matched.drop(columns=["position"], errors="ignore").merge(
            actuals[["name_key", "position", "actual_rank_total", "actual_rank_avg"]],
            on="name_key",
            how="left",
        )
        rank_scope = "full positional actual pool"
    else:
        # cleaned_actual_ranks.csv has no position column, so this is the rank
        # within the predicted-player pool rather than the full positional pool.
        matched["actual_rank_total"] = matched.groupby("position")["TTL"].rank(
            method="min", ascending=False
        )
        matched["actual_rank_avg"] = matched.groupby("position")["AVG"].rank(
            method="min", ascending=False
        )
        rank_scope = "predicted-player pool"

    for metric in ("total", "avg"):
        matched[f"rank_error_{metric}"] = (
            matched[f"actual_rank_{metric}"] - matched["predicted_rank"]
        )
        matched[f"absolute_error_{metric}"] = matched[f"rank_error_{metric}"].abs()
        matched[f"adp_rank_error_{metric}"] = (
            matched[f"actual_rank_{metric}"] - matched["adp_rank"]
        )
        matched[f"adp_absolute_error_{metric}"] = matched[
            f"adp_rank_error_{metric}"
        ].abs()
        matched[f"ecr_rank_error_{metric}"] = (
            matched[f"actual_rank_{metric}"] - matched["ecr_rank"]
        )
        matched[f"ecr_absolute_error_{metric}"] = matched[
            f"ecr_rank_error_{metric}"
        ].abs()
        matched[f"your_advantage_{metric}"] = (
            matched[f"adp_absolute_error_{metric}"]
            - matched[f"absolute_error_{metric}"]
        )
        matched[f"your_ecr_advantage_{metric}"] = (
            matched[f"ecr_absolute_error_{metric}"]
            - matched[f"absolute_error_{metric}"]
        )
    matched["market_disagreement"] = matched["adp_rank"] - matched["predicted_rank"]
    matched["actual_vs_adp_total"] = matched["adp_rank"] - matched["actual_rank_total"]
    matched["actual_vs_adp_avg"] = matched["adp_rank"] - matched["actual_rank_avg"]
    matched["rank_scope"] = rank_scope
    matched["adp_rank_scope"] = "predicted-player pool by position"
    return (
        matched,
        unmatched_predictions,
        duplicate_names,
        unmatched_adp,
        duplicate_adp_names,
        unmatched_ecr,
        duplicate_ecr_names,
    )


def make_rank_figure(
    data: pd.DataFrame,
    initial_metric: str,
    position: str,
    rank_column: str = "predicted_rank",
    rank_label: str = "Your",
) -> go.Figure:
    figure = go.Figure()
    position_data = data[data["position"].eq(position)].dropna(subset=[rank_column])
    max_rank = max(
        position_data[rank_column].max(),
        position_data[["actual_rank_total", "actual_rank_avg"]].max().max(),
    )
    cutoffs = [None, 12, 24, 36] + ([48] if position == "WR" else [])

    def visible_for_metric(metric: str) -> list[bool]:
        visibility = [False] * (len(cutoffs) * 4)
        metric_offset = 0 if metric == "total" else 2
        visibility[metric_offset : metric_offset + 2] = [True, True]
        return visibility

    def visible_for_cutoff(cutoff: int | None) -> list[bool]:
        return [
            cutoff_value == cutoff and cutoff_metric == initial_metric
            for cutoff_value in cutoffs
            for cutoff_metric in ("total", "avg")
            for _ in (0, 1)
        ]

    for cutoff in cutoffs:
        subset = (
            position_data
            if cutoff is None
            else position_data[position_data[rank_column].le(cutoff)]
        )
        for metric in ("total", "avg"):
            predicted_ranks = subset[rank_column]
            actual_ranks = subset[f"actual_rank_{metric}"]
            if predicted_ranks.nunique() > 1:
                slope = predicted_ranks.cov(actual_ranks) / predicted_ranks.var()
                intercept = actual_ranks.mean() - slope * predicted_ranks.mean()
            else:
                slope = 0
                intercept = actual_ranks.mean()
            custom_columns = [
                "player",
                "position",
                "TTL",
                "AVG",
                "GP",
                rank_column,
                f"actual_rank_{metric}",
            ]
            active = cutoff is None and metric == initial_metric
            figure.add_trace(
                go.Scatter(
                    x=subset[rank_column],
                    y=subset[f"actual_rank_{metric}"],
                    mode="markers",
                    name=position,
                    marker={
                        "color": POSITION_COLORS[position],
                        "size": 10,
                        "line": {"width": 1, "color": "white"},
                    },
                    customdata=subset[custom_columns],
                    hovertemplate=(
                        "<b>%{customdata[0]}</b> (%{customdata[1]})<br>"
                        f"{rank_label} rank: %{{customdata[5]:.0f}}<br>Actual rank: %{{customdata[6]:.0f}}<br>"
                        "Total: %{customdata[2]:.1f}<br>Avg: %{customdata[3]:.1f}<br>Games: %{customdata[4]:.0f}<extra></extra>"
                    ),
                    visible=active,
                )
            )
            figure.add_trace(
                go.Scatter(
                    x=[0, max_rank],
                    y=[intercept, slope * max_rank + intercept],
                    mode="lines",
                    name=f"Best fit ({metric})",
                    line={
                        "color": POSITION_COLORS[position],
                        "dash": "dot",
                        "width": 3,
                    },
                    hoverinfo="skip",
                    visible=active,
                )
            )
    figure.add_shape(
        type="line",
        x0=0,
        y0=0,
        x1=max_rank,
        y1=max_rank,
        line={"dash": "dash", "color": "#6b7280"},
    )
    figure.update_layout(
        title={
            "text": f"{position}: {rank_label.lower()} rank vs actual rank ({'total points' if initial_metric == 'total' else 'avg PPG'})"
        },
        xaxis={
            "title": {"text": f"{rank_label} positional rank"},
            "range": [0, max_rank],
            "constrain": "domain",
            "automargin": False,
        },
        yaxis={
            "title": {"text": ""},
            "range": [0, max_rank],
            "scaleanchor": "x",
            "scaleratio": 1,
            "constrain": "domain",
            "automargin": False,
        },
        template="plotly_white",
        width=480,
        height=480,
        margin={"t": 70, "b": 65, "l": 60, "r": 20},
        legend={
            "orientation": "h",
            "x": 0,
            "y": -0.14,
            "xanchor": "left",
            "yanchor": "top",
        },
    )
    return figure


def make_accuracy_figure(data: pd.DataFrame, initial_metric: str) -> go.Figure:
    figure = go.Figure()
    metric_maxima = []
    for metric in ("total", "avg"):
        summary = (
            data.groupby("position")[f"absolute_error_{metric}"]
            .median()
            .reindex(POSITIONS)
        )
        metric_maxima.append(summary.max())
        figure.add_trace(
            go.Bar(
                x=list(POSITIONS),
                y=summary,
                name="Median absolute error",
                visible=metric == initial_metric,
                marker_color=[POSITION_COLORS[position] for position in POSITIONS],
            )
        )
    accuracy_axis_max = max(metric_maxima) * 1.1
    figure.update_layout(
        title="Median absolute rank error by position",
        yaxis_title="Median rank error (lower is better)",
        yaxis={"range": [0, accuracy_axis_max]},
        template="plotly_white",
        height=450,
        margin={"t": 140},
        updatemenus=[
            {
                "type": "buttons",
                "direction": "right",
                "x": 0,
                "y": 1.08,
                "buttons": [
                    {
                        "label": "Total points",
                        "method": "update",
                        "args": [
                            {"visible": [True, False]},
                            {
                                "yaxis": {
                                    "range": [0, accuracy_axis_max],
                                    "title": {
                                        "text": "Median rank error (lower is better)"
                                    },
                                },
                                "title": {
                                    "text": "Median absolute rank error by position: total points"
                                },
                            },
                        ],
                    },
                    {
                        "label": "Avg PPG",
                        "method": "update",
                        "args": [
                            {"visible": [False, True]},
                            {
                                "yaxis": {
                                    "range": [0, accuracy_axis_max],
                                    "title": {
                                        "text": "Median rank error (lower is better)"
                                    },
                                },
                                "title": {
                                    "text": "Median absolute rank error by position: avg PPG"
                                },
                            },
                        ],
                    },
                ],
            }
        ],
    )
    return figure


def make_adp_actual_figure(data: pd.DataFrame, initial_metric: str) -> go.Figure:
    figure = go.Figure()
    adp_data = data.dropna(subset=["adp_rank"]).copy()
    max_rank = max(
        adp_data["adp_rank"].max(),
        adp_data[["actual_rank_total", "actual_rank_avg"]].max().max(),
    )
    for metric in ("total", "avg"):
        for position in POSITIONS:
            subset = adp_data[adp_data["position"].eq(position)]
            figure.add_trace(
                go.Scatter(
                    x=subset["adp_rank"],
                    y=subset[f"actual_rank_{metric}"],
                    mode="markers",
                    name=position,
                    legendgroup=position,
                    marker={"color": POSITION_COLORS[position], "size": 9},
                    customdata=subset[
                        [
                            "player",
                            "position",
                            "adp_rank",
                            f"actual_rank_{metric}",
                            "TTL",
                            "AVG",
                            "GP",
                        ]
                    ],
                    hovertemplate=(
                        "<b>%{customdata[0]}</b> (%{customdata[1]})<br>"
                        "ADP positional rank: %{customdata[2]:.0f}<br>Actual rank: %{customdata[3]:.0f}<br>"
                        "Total: %{customdata[4]:.1f}<br>Avg: %{customdata[5]:.1f}<br>Games: %{customdata[6]:.0f}<extra></extra>"
                    ),
                    visible=metric == initial_metric,
                )
            )
    figure.add_shape(
        type="line",
        x0=0,
        y0=0,
        x1=max_rank,
        y1=max_rank,
        line={"dash": "dash", "color": "#6b7280"},
    )
    figure.update_layout(
        title={
            "text": f"ADP versus actual rank ({'total points' if initial_metric == 'total' else 'avg PPG'})"
        },
        xaxis={"title": "ADP positional rank", "range": [0, max_rank]},
        yaxis={
            "title": "Actual rank",
            "range": [0, max_rank],
            "scaleanchor": "x",
            "scaleratio": 1,
        },
        template="plotly_white",
        autosize=True,
        width=None,
        height=700,
        margin={"t": 120},
        updatemenus=[
            {
                "type": "buttons",
                "direction": "right",
                "x": 0,
                "y": 1.08,
                "buttons": [
                    {
                        "label": "Total points",
                        "method": "update",
                        "args": [
                            {"visible": [True, False]},
                            {
                                "xaxis": {"range": [0, max_rank]},
                                "yaxis": {
                                    "range": [0, max_rank],
                                    "title": {"text": "Actual rank"},
                                },
                                "title": {
                                    "text": "ADP versus actual rank (total points)"
                                },
                            },
                        ],
                    },
                    {
                        "label": "Avg PPG",
                        "method": "update",
                        "args": [
                            {
                                "visible": [False] * len(POSITIONS)
                                + [True] * len(POSITIONS)
                            },
                            {
                                "xaxis": {"range": [0, max_rank]},
                                "yaxis": {
                                    "range": [0, max_rank],
                                    "title": {"text": "Actual rank"},
                                },
                                "title": {"text": "ADP versus actual rank (avg PPG)"},
                            },
                        ],
                    },
                ],
            }
        ],
    )
    return figure


def make_value_figure(
    data: pd.DataFrame,
    initial_metric: str,
    position: str,
    market_rank_column: str = "adp_rank",
    market_label: str = "ADP",
) -> go.Figure:
    figure = go.Figure()
    market_data = (
        data[data["position"].eq(position)].dropna(subset=[market_rank_column]).copy()
    )
    market_data["market_disagreement"] = (
        market_data[market_rank_column] - market_data["predicted_rank"]
    )
    max_x = max(market_data["market_disagreement"].abs().max(), 1)
    max_y = max(
        pd.concat(
            [
                market_data[market_rank_column] - market_data["actual_rank_total"],
                market_data[market_rank_column] - market_data["actual_rank_avg"],
            ],
            axis=1,
        )
        .abs()
        .max()
        .max(),
        1,
    )
    axis_x = max_x * 1.15
    axis_y = max_y * 1.15

    cutoffs = [None, 12, 24, 36] + ([48] if position == "WR" else [])

    def visible_for_metric(metric: str) -> list[bool]:
        return [
            cutoff_metric == metric
            for cutoff in cutoffs
            for cutoff_metric in ("total", "avg")
        ]

    def visible_for_cutoff(cutoff: int | None) -> list[bool]:
        return [
            cutoff_value == cutoff and cutoff_metric == initial_metric
            for cutoff_value in cutoffs
            for cutoff_metric in ("total", "avg")
        ]

    for cutoff in cutoffs:
        subset = (
            market_data
            if cutoff is None
            else market_data[market_data["predicted_rank"].le(cutoff)]
        )
        for metric in ("total", "avg"):
            actual_column = f"actual_rank_{metric}"
            subset = subset.copy()
            subset["realized_disagreement"] = (
                subset[market_rank_column] - subset[actual_column]
            )
            figure.add_trace(
                go.Scatter(
                    x=subset["market_disagreement"],
                    y=subset["realized_disagreement"],
                    mode="markers",
                    name=position,
                    marker={"color": POSITION_COLORS[position], "size": 9},
                    customdata=subset[
                        [
                            "player",
                            "position",
                            "predicted_rank",
                            market_rank_column,
                            "actual_rank_total",
                            "actual_rank_avg",
                            "market_disagreement",
                        ]
                    ],
                    hovertemplate=(
                        "<b>%{customdata[0]}</b> (%{customdata[1]})<br>"
                        f"Your rank: %{{customdata[2]:.0f}}<br>{market_label} rank: %{{customdata[3]:.0f}}<br>"
                        "Actual rank: %{customdata[4]:.0f} total / %{customdata[5]:.0f} avg<br>"
                        f"Your rank minus {market_label}: %{{customdata[6]:.1f}}<extra></extra>"
                    ),
                    visible=cutoff is None and metric == initial_metric,
                )
            )
    """
    for metric in ("total", "avg"):
        figure.add_trace(
            go.Scatter(
                x=adp_data["market_disagreement"],
                y=adp_data[f"actual_vs_adp_{metric}"],
                mode="markers",
                name=position,
                marker={"color": POSITION_COLORS[position], "size": 9},
                customdata=adp_data[
                    [
                        "player",
                        "position",
                        "predicted_rank",
                        "adp_rank",
                        "actual_rank_total",
                        "actual_rank_avg",
                        "market_disagreement",
                    ]
                ],
                hovertemplate=(
                    "<b>%{customdata[0]}</b> (%{customdata[1]})<br>"
                    "Your rank: %{customdata[2]:.0f}<br>ADP rank: %{customdata[3]:.0f}<br>"
                    "Actual rank: %{customdata[4]:.0f} total / %{customdata[5]:.0f} avg<br>"
                    "Your rank minus market: %{customdata[6]:.1f}<extra></extra>"
                ),
                visible=metric == initial_metric,
            )
        )
    """
    figure.add_shape(
        type="line",
        x0=0,
        y0=-axis_y,
        x1=0,
        y1=axis_y,
        line={"dash": "dash", "color": "#9ca3af"},
    )
    figure.add_shape(
        type="line",
        x0=-axis_x,
        y0=0,
        x1=axis_x,
        y1=0,
        line={"dash": "dash", "color": "#9ca3af"},
    )
    figure.add_annotation(
        x=-axis_x * 0.55,
        y=axis_y * 0.82,
        text="Missed value",
        showarrow=False,
        font={"color": "#6b7280"},
    )
    figure.add_annotation(
        x=axis_x * 0.55,
        y=axis_y * 0.82,
        text="Good call",
        showarrow=False,
        font={"color": "#059669"},
    )
    figure.add_annotation(
        x=-axis_x * 0.55,
        y=-axis_y * 0.82,
        text="Good fade",
        showarrow=False,
        font={"color": "#2563eb"},
    )
    figure.add_annotation(
        x=axis_x * 0.55,
        y=-axis_y * 0.82,
        text="Bad reach",
        showarrow=False,
        font={"color": "#dc2626"},
    )
    figure.update_layout(
        title={
            "text": f"{position}: {market_label} disagreement and realized value ({initial_metric})"
        },
        xaxis={
            "title": {"text": f"Positive = you ranked earlier than {market_label}"},
            "range": [-axis_x, axis_x],
        },
        yaxis={
            "title": {"text": "Positive = player finished better than ADP"},
            "range": [-axis_y, axis_y],
        },
        template="plotly_white",
        height=600,
        margin={"t": 120},
        updatemenus=[
            {
                "type": "buttons",
                "direction": "right",
                "x": 0,
                "y": 1.08,
                "buttons": [
                    {
                        "label": "Total points",
                        "method": "update",
                        "args": [
                            {"visible": visible_for_metric("total")},
                            {
                                "yaxis": {
                                    "title": {
                                        "text": f"Positive = player finished better than {market_label}"
                                    },
                                    "range": [-axis_y, axis_y],
                                },
                                "xaxis": {
                                    "title": {
                                        "text": f"Positive = you ranked earlier than {market_label}"
                                    },
                                    "range": [-axis_x, axis_x],
                                },
                                "title": {
                                    "text": f"{position}: {market_label} disagreement and realized value (total points)"
                                },
                            },
                        ],
                    },
                    {
                        "label": "Avg PPG",
                        "method": "update",
                        "args": [
                            {"visible": visible_for_metric("avg")},
                            {
                                "yaxis": {
                                    "title": {
                                        "text": f"Positive = player finished better than {market_label}"
                                    },
                                    "range": [-axis_y, axis_y],
                                },
                                "xaxis": {
                                    "title": {
                                        "text": f"Positive = you ranked earlier than {market_label}"
                                    },
                                    "range": [-axis_x, axis_x],
                                },
                                "title": {
                                    "text": f"{position}: {market_label} disagreement and realized value (avg PPG)"
                                },
                            },
                        ],
                    },
                ],
            },
            {
                "type": "dropdown",
                "direction": "down",
                "x": 0.42,
                "y": 1.08,
                "buttons": [
                    {
                        "label": "ALL",
                        "method": "update",
                        "args": [{"visible": visible_for_cutoff(None)}],
                    },
                    {
                        "label": "Top 12",
                        "method": "update",
                        "args": [{"visible": visible_for_cutoff(12)}],
                    },
                    {
                        "label": "Top 24",
                        "method": "update",
                        "args": [{"visible": visible_for_cutoff(24)}],
                    },
                    {
                        "label": "Top 36",
                        "method": "update",
                        "args": [{"visible": visible_for_cutoff(36)}],
                    },
                    *(
                        [
                            {
                                "label": "Top 48",
                                "method": "update",
                                "args": [{"visible": visible_for_cutoff(48)}],
                            }
                        ]
                        if position == "WR"
                        else []
                    ),
                ],
            },
        ],
    )
    return figure


def make_scoreboard_figure(data: pd.DataFrame, initial_metric: str) -> go.Figure:
    figure = go.Figure()
    adp_data = data.dropna(subset=["adp_rank"])
    for metric in ("total", "avg"):
        for source, label, color in (
            ("absolute_error", "You", "#111827"),
            ("adp_absolute_error", "ADP", "#9ca3af"),
        ):
            summary = (
                adp_data.groupby("position")[f"{source}_{metric}"]
                .median()
                .reindex(POSITIONS)
            )
            figure.add_trace(
                go.Bar(
                    x=list(POSITIONS),
                    y=summary,
                    name=label,
                    marker_color=color,
                    visible=metric == initial_metric,
                )
            )
    figure.update_layout(
        title={"text": "Ranking accuracy: you versus ADP"},
        yaxis_title="Median absolute rank error (lower is better)",
        barmode="group",
        template="plotly_white",
        height=500,
        margin={"t": 120},
        updatemenus=[
            {
                "type": "buttons",
                "direction": "right",
                "x": 0,
                "y": 1.08,
                "buttons": [
                    {
                        "label": "Total points",
                        "method": "update",
                        "args": [
                            {"visible": [True, True, False, False]},
                            {
                                "yaxis": {
                                    "title": {
                                        "text": "Median absolute rank error (lower is better)"
                                    }
                                },
                                "title": {
                                    "text": "Ranking accuracy: you versus ADP (total points)"
                                },
                            },
                        ],
                    },
                    {
                        "label": "Avg PPG",
                        "method": "update",
                        "args": [
                            {"visible": [False, False, True, True]},
                            {
                                "yaxis": {
                                    "title": {
                                        "text": "Median absolute rank error (lower is better)"
                                    }
                                },
                                "title": {
                                    "text": "Ranking accuracy: you versus ADP (avg PPG)"
                                },
                            },
                        ],
                    },
                ],
            }
        ],
    )
    return figure


def make_html(
    data: pd.DataFrame,
    output: str,
    initial_metric: str,
    min_games: int,
    unmatched: list[str],
    duplicates: list[str],
    unmatched_adp: list[str],
    duplicate_adp_names: list[str],
    unmatched_ecr: list[str],
    duplicate_ecr_names: list[str],
) -> None:
    rank_figures = [
        make_rank_figure(data, initial_metric, position) for position in POSITIONS
    ]
    adp_rank_figures = [
        make_rank_figure(
            data,
            initial_metric,
            position,
            rank_column="adp_rank",
            rank_label="ADP",
        )
        for position in POSITIONS
    ]
    ecr_rank_figures = [
        make_rank_figure(
            data,
            initial_metric,
            position,
            rank_column="ecr_rank",
            rank_label="ECR",
        )
        for position in POSITIONS
    ]
    value_figures = [
        make_value_figure(data, initial_metric, position) for position in POSITIONS
    ]
    ecr_value_figures = [
        make_value_figure(
            data,
            initial_metric,
            position,
            market_rank_column="ecr_rank",
            market_label="ECR",
        )
        for position in POSITIONS
    ]
    adp_data = data.dropna(subset=["adp_rank"]).copy()
    display_columns = [
        "player",
        "position",
        "predicted_rank",
        "adp_rank",
        "actual_rank_total",
        "actual_rank_avg",
        "rank_error_total",
        "rank_error_avg",
        "adp_absolute_error_total",
        "adp_absolute_error_avg",
        "your_advantage_total",
        "TTL",
        "AVG",
        "GP",
    ]
    table_data = adp_data.sort_values("your_advantage_total", ascending=False)[
        display_columns
    ].copy()
    table_data.columns = [
        "Player",
        "Pos",
        "Predicted",
        "ADP pos.",
        "Actual total",
        "Actual avg",
        "Error total",
        "Error avg",
        "ADP error total",
        "ADP error avg",
        "Your advantage",
        "Total",
        "Avg/G",
        "GP",
    ]
    numeric_columns = [
        column for column in table_data.columns if column not in ("Player", "Pos")
    ]
    table_data[numeric_columns] = table_data[numeric_columns].round(1)
    table = go.Figure(
        data=[
            go.Table(
                header={
                    "values": list(table_data.columns),
                    "fill_color": "#1f2937",
                    "font": {"color": "white"},
                    "align": "left",
                },
                cells={
                    "values": [table_data[column] for column in table_data.columns],
                    "align": "left",
                    "height": 25,
                },
            )
        ]
    )
    table.update_layout(
        title="Player comparison (sorted by total-rank error)", height=700
    )

    metrics = {
        "matched": len(data),
        "adp_matched": len(adp_data),
    }
    rank_scope = data["rank_scope"].iloc[0] if not data.empty else "unknown"
    audit_lines = [
        f"Matched players after the games-played filter: {len(data)}",
        f"Minimum games: {min_games}",
        f"Actual rank scope: {rank_scope}",
        f"Unmatched prediction names: {', '.join(unmatched) if unmatched else 'None'}",
        f"Duplicate actual names excluded: {', '.join(duplicates) if duplicates else 'None'}",
        f"ADP positional rank scope: predicted-player pool by position",
        f"Predictions missing from ADP: {', '.join(unmatched_adp) if unmatched_adp else 'None'}",
        f"Duplicate ADP names excluded: {', '.join(duplicate_adp_names) if duplicate_adp_names else 'None'}",
        f"Predictions missing from ECR: {', '.join(unmatched_ecr) if unmatched_ecr else 'None'}",
        f"Duplicate ECR names excluded: {', '.join(duplicate_ecr_names) if duplicate_ecr_names else 'None'}",
    ]

    def format_metric(value: float) -> str:
        return f"{value:.2f}" if pd.notna(value) else "n/a"

    def render_comparison_list(
        source_data: pd.DataFrame,
        position: str,
        source_label: str,
        advantage_column: str,
        source_rank_column: str,
        source_error_column: str,
        metric: str,
        your_wins: bool,
    ) -> str:
        position_data = source_data[source_data["position"].eq(position)].copy()
        position_data = position_data.dropna(
            subset=[advantage_column, source_rank_column, source_error_column]
        )
        if your_wins:
            position_data = position_data[position_data[advantage_column].gt(0)]
            heading = f"Your top 5 over {source_label}"
            position_data = position_data.sort_values(advantage_column, ascending=False)
        else:
            position_data = position_data[position_data[advantage_column].lt(0)]
            heading = f"{source_label} top 5 over you"
            position_data = position_data.sort_values(advantage_column, ascending=True)
        position_data = position_data.head(5)
        rows = "".join(
            "<tr>"
            f"<td>{row.player}</td><td>{row.predicted_rank:.0f}</td>"
            f"<td>{row[source_rank_column]:.0f}</td>"
            f"<td>{row[f'actual_rank_{metric}']:.0f}</td>"
            f"<td>{abs(row[f'absolute_error_{metric}']):.1f}</td>"
            f"<td>{abs(row[source_error_column]):.1f}</td>"
            f"<td>{abs(row[advantage_column]):.1f}</td>"
            "</tr>"
            for _, row in position_data.iterrows()
        )
        if not rows:
            rows = "<tr><td colspan='7'>None</td></tr>"
        return (
            f"<div class='comparison-list {'comparison-win' if your_wins else 'comparison-loss'}'><h4>{position}: {heading}</h4>"
            "<table><thead><tr><th>Player</th><th>Your</th>"
            f"<th>{source_label}</th><th>Actual</th><th>Your error</th>"
            f"<th>{source_label} error</th><th>Margin</th></tr></thead>"
            f"<tbody>{rows}</tbody></table></div>"
        )

    def render_comparison_tables(metric: str) -> str:
        comparison_sections = []
        for position in POSITIONS:
            comparison_sections.extend(
                [
                    render_comparison_list(
                        data,
                        position,
                        "ADP",
                        f"your_advantage_{metric}",
                        "adp_rank",
                        f"adp_rank_error_{metric}",
                        metric,
                        True,
                    ),
                    render_comparison_list(
                        data,
                        position,
                        "ADP",
                        f"your_advantage_{metric}",
                        "adp_rank",
                        f"adp_rank_error_{metric}",
                        metric,
                        False,
                    ),
                    render_comparison_list(
                        data,
                        position,
                        "ECR",
                        f"your_ecr_advantage_{metric}",
                        "ecr_rank",
                        f"ecr_rank_error_{metric}",
                        metric,
                        True,
                    ),
                    render_comparison_list(
                        data,
                        position,
                        "ECR",
                        f"your_ecr_advantage_{metric}",
                        "ecr_rank",
                        f"ecr_rank_error_{metric}",
                        metric,
                        False,
                    ),
                ]
            )
        return "".join(comparison_sections)

    comparison_html = (
        f"<div class='comparison-metric' data-metric='total'>{render_comparison_tables('total')}</div>"
        f"<div class='comparison-metric' data-metric='avg'>{render_comparison_tables('avg')}</div>"
    )

    def render_rank_pair(
        index: int,
        position: str,
        personal: go.Figure,
        market: go.Figure,
        ecr: go.Figure,
    ) -> str:
        top_48_option = "<option value='48'>Top 48</option>" if position == "WR" else ""
        return (
            f"<div class='rank-pair' data-position='{position}'>"
            "<div class='rank-controls'>"
            "<label>Metric <select class='metric-select'>"
            f"<option value='total'{' selected' if initial_metric == 'total' else ''}>Total points</option>"
            f"<option value='avg'{' selected' if initial_metric == 'avg' else ''}>Avg PPG</option>"
            "</select></label>"
            "<label>Scope <select class='scope-select'>"
            "<option value='all' selected>ALL</option>"
            "<option value='12'>Top 12</option>"
            "<option value='24'>Top 24</option>"
            "<option value='36'>Top 36</option>"
            f"{top_48_option}</select></label>"
            "</div><div class='rank-row'>"
            f"<section>{personal.to_html(full_html=False, include_plotlyjs='cdn' if index == 0 else False, config={'responsive': True})}</section>"
            f"<section>{market.to_html(full_html=False, include_plotlyjs=False, config={'responsive': True})}</section>"
            f"<section>{ecr.to_html(full_html=False, include_plotlyjs=False, config={'responsive': True})}</section>"
            "</div></div>"
        )

    rank_charts_html = "".join(
        render_rank_pair(index, position, personal, market, ecr)
        for index, (position, personal, market, ecr) in enumerate(
            zip(POSITIONS, rank_figures, adp_rank_figures, ecr_rank_figures)
        )
    )
    rank_sync_script = """
    <script>
    document.querySelectorAll('.rank-pair').forEach(function (pair) {
        const charts = pair.querySelectorAll('.js-plotly-plot');
        const metricSelect = pair.querySelector('.metric-select');
        const scopeSelect = pair.querySelector('.scope-select');
        const position = pair.dataset.position;
        const sourceLabels = ['Your', 'ADP', 'ECR'];
        const cutoffs = position === 'WR' ? ['all', '12', '24', '36', '48'] : ['all', '12', '24', '36'];
        pair.dataset.scope = scopeSelect.value;

        function updateCharts() {
            const metric = metricSelect.value;
            const selectedScope = pair.dataset.scope;
            const cutoffIndex = cutoffs.indexOf(selectedScope);
            const metricOffset = metric === 'total' ? 0 : 2;
            charts.forEach(function (chart, chartIndex) {
                const visible = Array(chart.data.length).fill(false);
                const traceIndex = cutoffIndex * 4 + metricOffset;
                visible[traceIndex] = true;
                visible[traceIndex + 1] = true;
                Plotly.update(chart, {visible: visible});
                Plotly.relayout(chart, {
                    'title.text': position + ': ' + sourceLabels[chartIndex].toLowerCase() +
                        ' rank vs actual rank (' +
                        (metric === 'total' ? 'total points' : 'avg PPG') + ')'
                });
            });
            if (scopeSelect.value !== selectedScope) {
                scopeSelect.value = selectedScope;
            }
        }

        metricSelect.addEventListener('change', updateCharts);
        scopeSelect.addEventListener('change', function () {
            pair.dataset.scope = scopeSelect.value;
            updateCharts();
        });
    });
    </script>
    """
    comparison_sync_script = """
    <script>
    const comparisonMetricSelect = document.querySelector('.comparison-metric-select');
    const comparisonMetricPanels = document.querySelectorAll('.comparison-metric');
    function updateComparisonTables() {
        const metric = comparisonMetricSelect.value;
        comparisonMetricPanels.forEach(function (panel) {
            panel.hidden = panel.dataset.metric !== metric;
        });
    }
    comparisonMetricSelect.addEventListener('change', updateComparisonTables);
    updateComparisonTables();
    </script>
    """
    adp_charts_html = "".join(
        f"<section>{figure.to_html(full_html=False, include_plotlyjs=False, config={'responsive': True})}</section>"
        for figure in value_figures
    )
    ecr_charts_html = "".join(
        f"<section>{figure.to_html(full_html=False, include_plotlyjs=False, config={'responsive': True})}</section>"
        for figure in ecr_value_figures
    )
    html = f"""
    <html><head><meta charset='utf-8'><title>Fantasy Ranking Analysis</title>
    <style>body{{font-family:Arial,sans-serif;max-width:1500px;margin:30px auto;padding:0 20px;color:#1f2937;overflow-x:hidden}}h1{{margin-bottom:4px}}.subtitle{{color:#6b7280}}.metrics{{display:flex;gap:16px;flex-wrap:wrap;margin:24px 0}}.metric{{background:#f3f4f6;padding:16px 22px;border-radius:8px;min-width:150px}}.metric strong{{display:block;font-size:26px}}.source-charts,.market-charts{{display:block}}.market-charts{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:24px;justify-content:center}}.rank-pair{{display:block;margin-bottom:32px}}.rank-controls{{display:flex;gap:16px;align-items:center;margin:0 0 10px}}.rank-controls label{{display:flex;gap:6px;align-items:center;font-size:14px}}.rank-controls select{{font:inherit;padding:4px 8px}}.rank-row{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:24px}}.source-charts section,.market-charts section{{margin:0;min-width:0;overflow:hidden}}.plotly-graph-div{{width:100%!important;max-width:100%!important}}.comparison-controls{{display:flex;align-items:center;gap:8px;margin:12px 0 16px}}.comparison-controls select{{font:inherit;padding:5px 8px}}.comparison-grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}}.comparison-metric[hidden]{{display:none}}.comparison-list{{border:1px solid #dbe4ec;border-radius:6px;padding:12px;min-width:0}}.comparison-list h4{{margin:0 0 8px;font-size:15px}}.comparison-list table{{width:100%;border-collapse:collapse;font-size:13px}}.comparison-list th,.comparison-list td{{padding:5px 6px;border-bottom:1px solid #dbe4ec;text-align:left}}.comparison-list th{{background:#f3f4f6;font-weight:600}}.comparison-win{{background:#f0f9ff;border-color:#bae6fd}}.comparison-loss{{background:#fff7f7;border-color:#fecaca}}.comparison-win th{{background:#e0f2fe}}.comparison-loss th{{background:#fee2e2}}.comparison-win td:last-child{{color:#0369a1;font-weight:700}}.comparison-loss td:last-child{{color:#b91c1c;font-weight:700}}.audit{{background:#fff7ed;border-left:4px solid #f97316;padding:12px 18px;line-height:1.7}}section{{margin:34px 0}}@media (max-width:1100px){{.rank-row,.market-charts,.comparison-grid{{grid-template-columns:1fr}}}}</style></head><body>
    <h1>2025 Fantasy Ranking Analysis</h1><div class='subtitle'>Use the controls above each position to compare sources, metrics, and ranking scopes.</div>
    <div class='metrics'><div class='metric'><strong>{metrics['matched']}</strong>Matched players</div><div class='metric'><strong>{metrics['adp_matched']}</strong>ADP matches</div></div>
    <div class='source-charts'>{rank_charts_html}</div>{rank_sync_script}
    <h2>Market analysis: ADP</h2><div class='market-charts'>{adp_charts_html}</div>
    <h2>Market analysis: ECR</h2><div class='market-charts'>{ecr_charts_html}</div>
    <section><h2>Top-five comparisons</h2><p>Lower rank error is better. Margin is the difference in absolute rank error.</p><div class='comparison-controls'><label for='comparison-metric-select'>Metric</label><select id='comparison-metric-select' class='comparison-metric-select'><option value='total'{' selected' if initial_metric == 'total' else ''}>Total points</option><option value='avg'{' selected' if initial_metric == 'avg' else ''}>Avg PPG</option></select></div><div class='comparison-grid'>{comparison_html}</div></section>{comparison_sync_script}
    <section><h2>Join audit</h2><div class='audit'>{'<br>'.join(audit_lines)}</div></section>
    </body></html>
    """
    Path(output).write_text(html, encoding="utf-8")


def main() -> None:
    args = parse_args()
    predictions = load_predictions(args.predictions)
    actuals = load_actuals(args.actual)
    adp = load_adp(args.adp)
    ecr = load_ecr(args.ecr)
    (
        data,
        unmatched,
        duplicates,
        unmatched_adp,
        duplicate_adp_names,
        unmatched_ecr,
        duplicate_ecr_names,
    ) = calculate_analysis(predictions, actuals, adp, ecr, args.min_games)
    if data.empty:
        raise ValueError(
            "No matched players remain after applying the minimum-games filter."
        )
    make_html(
        data,
        args.output,
        args.metric,
        args.min_games,
        unmatched,
        duplicates,
        unmatched_adp,
        duplicate_adp_names,
        unmatched_ecr,
        duplicate_ecr_names,
    )
    print(f"Wrote {args.output} with {len(data)} matched players.")
    if unmatched:
        print("Unmatched prediction names:", ", ".join(unmatched))
    if duplicates:
        print("Suspicious duplicate actual names excluded:", ", ".join(duplicates))
    if unmatched_adp:
        print("Predictions missing from ADP:", ", ".join(unmatched_adp))
    if duplicate_adp_names:
        print(
            "Suspicious duplicate ADP names excluded:", ", ".join(duplicate_adp_names)
        )
    if unmatched_ecr:
        print("Predictions missing from ECR:", ", ".join(unmatched_ecr))
    if duplicate_ecr_names:
        print(
            "Suspicious duplicate ECR names excluded:", ", ".join(duplicate_ecr_names)
        )


if __name__ == "__main__":
    main()
