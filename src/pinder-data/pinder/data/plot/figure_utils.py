from __future__ import annotations
from copy import deepcopy
from pathlib import Path
from time import sleep
from typing import Any

import plotly.express as px
from plotly.graph_objs._figure import Figure

from pinder.core.utils.log import setup_logger

log = setup_logger(__name__)


def add_xaxis_title(
    fig: Figure, x_title: str, x: float = 0.5, y: float = -0.1
) -> Figure:
    fig = fig.add_annotation(
        x=x,
        y=y,
        text=x_title,
        font=dict(size=26, family="Helvetica"),
        xref="paper",
        yref="paper",
        showarrow=False,
    )
    return fig


def add_yaxis_title(
    fig: Figure, y_title: str, x: float = -0.1, y: float = 0.5
) -> Figure:
    fig = fig.add_annotation(
        x=x,
        y=y,
        text=y_title,
        font=dict(size=26, family="Helvetica"),
        xref="paper",
        yref="paper",
        textangle=-90,
        showarrow=False,
    )
    return fig


def bold_annotations(fig: Figure) -> Figure:
    return fig.for_each_annotation(
        lambda a: a.update(text="<b>" + a.text.split("=")[-1] + "</b>")
    )


def bold_trace_name(fig: Figure) -> Figure:
    return fig.for_each_trace(lambda t: t.update(name="<b>" + t.name + "</b>"))


def format_text_template(
    fig: Figure,
    marker_line_color: str = "#333",
    text_template: str = "%{text:.1f}%",
) -> Figure:
    return fig.update_traces(
        texttemplate=text_template,
        textfont=dict(size=20),
        marker_line=dict(width=2, color=marker_line_color),
    )


def remove_xaxis_titles(fig: Figure) -> Figure:
    return fig.for_each_xaxis(lambda x: x.update(title=""))


def remove_yaxis_titles(fig: Figure) -> Figure:
    return fig.for_each_yaxis(lambda y: y.update(title=""))


def remove_legend_title(fig: Figure) -> Figure:
    fig.layout["legend"]["title"]["text"] = ""
    return fig


def update_layout(
    fig: Figure,
    font: dict[str, int | str] = dict(size=26, family="Helvetica"),
    margin: dict[str, int] = dict(t=50, b=20, r=50, l=50),
) -> Figure:
    return fig.update_layout(
        font=font,
        margin=margin,
    )


def update_axes(
    fig: Figure,
    axis: str,
    linecolor: str,
    tickcolor: str,
    titlecolor: str,
    **kwargs: Any,
) -> Figure:
    yaxis_defaults = dict(
        showgrid=False,
        showline=True,  # add line at x=0
        showticklabels=True,
        linecolor=linecolor,  # line color
        linewidth=2.4,  # line size
        ticks="outside",  # ticks outside axis
        tickwidth=2.0,  # tick width
        tickcolor=tickcolor,  # tick color
        mirror=False,  # add ticks to top/right axes
        title_text="",
    )
    xaxis_defaults = dict(
        showgrid=True,
        showline=True,
        showticklabels=True,
        linecolor=linecolor,
        linewidth=2.4,
        mirror=False,
        title_text="",
        tickcolor=tickcolor,  # tick color
        color=tickcolor,
        title_font={"color": titlecolor},
    )
    if axis == "x":
        func = fig.update_xaxes
        axis_config = deepcopy(xaxis_defaults)
    else:
        func = fig.update_yaxes
        axis_config = deepcopy(yaxis_defaults)
    axis_config.update(kwargs)
    fig = func(**axis_config)
    return fig


def write_fig(fig: Figure, output_file: Path, scale: int = 5) -> None:
    if not isinstance(output_file, Path):
        output_file = Path(output_file)
    if not output_file.parent.is_dir():
        output_file.parent.mkdir(parents=True)
    if output_file.suffix == ".pdf":
        # Known issue where PDF render results in "Loading MathJax" box to be
        # included in the final render. Only happens on initial figure write:
        # https://github.com/plotly/plotly.py/issues/3469
        dummy_fig = px.scatter(x=[0, 1, 2, 3, 4], y=[0, 1, 4, 9, 16])
        dummy_fig.write_image(output_file)
        sleep(2)
    fig.write_image(output_file, scale=scale)
    log.debug(f"Wrote figure to {output_file}...")
