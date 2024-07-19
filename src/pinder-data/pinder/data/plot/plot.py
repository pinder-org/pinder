from __future__ import annotations
from dataclasses import dataclass

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from pinder.data.plot import figure_utils as futils


@dataclass
class Colors:
    # Neutral
    white: str = "#FFFFFF"
    gray: str = "#A6ADCB"

    # Pinks
    pink: str = "#e7bade"
    brightpink: str = "#FD4ECB"
    pinder: str = "#f26b6c"

    # Purples
    palepurple: str = "#c1c7e4"
    whitepurple: str = "#d6d3db"
    purple: str = "#8268F4"
    redpurple: str = "#CC79A7"

    # Greens
    green: str = "#A3D6B3"
    palegreen: str = "#5F9A8D"
    teal: str = "#38EADC"
    bluish_green: str = "#009E73"

    # Blues
    blue: str = "#b2dff6"
    cb_blue: str = "#0072B2"
    cb_skyblue: str = "#56B4E9"

    ice: str = "#AAD2E8"
    deepice: str = "#77B9F1"
    paleblue: str = "#61A4BB"
    denim: str = "#285290"
    navy: str = "#132747"
    midnight: str = "#0B131B"

    # Orange/yellow
    cb_orange: str = "#E69F00"
    cb_vermillion: str = "#D55E00"
    cb_yellow: str = "#F0E442"

    def palette(self, name: str, n: int | None = None) -> list[str]:
        all_colors = self.colors
        pal: list[str] = getattr(self, name)
        if n is not None and len(pal) < n:
            print(
                f"Warning: {name} palette has {len(pal)} colors, but {n} requested. Replacing with alternatives"
            )
            pal.extend([c for c in all_colors if c not in pal])
        if n is None:
            n = len(pal)
        return pal[0:n]

    def _repr_html_(self) -> None:
        palette_names = [
            prop
            for prop in dir(self)
            if not prop.startswith("_")
            and prop not in ["palette", "colors"]
            and prop not in self.__dict__.keys()
        ]
        vbar = []
        color_names = {v: k for k, v in self.__dict__.items()}
        for palette in palette_names:
            colors = getattr(self, palette)
            for c in colors:
                vbar.append(
                    {
                        "Color": color_names[c],
                        "hex": c,
                        "palette": palette,
                        "y": 100 / len(colors),
                        "palette_size": len(colors),
                    }
                )

        vbar_df = pd.DataFrame(vbar)
        palette_order = list(
            vbar_df.sort_values("palette_size", ascending=False)
            .drop_duplicates("palette")
            .palette
        )

        fig = px.bar(
            vbar_df,
            y="palette",
            x="y",
            color="Color",
            color_discrete_map=self.__dict__,
            category_orders={"palette": palette_order},
            template="simple_white",
            height=650,
            width=850,
            text="hex",
        )
        fig.show()

        fig = px.bar(
            vbar_df.sort_values("palette_size"),
            y="palette",
            x="y",
            color="Color",
            color_discrete_map=self.__dict__,
            category_orders={"palette": palette_order},
            template="plotly_dark",
            height=650,
            width=850,
            text="hex",
        )
        fig.show()

    @property
    def colors(self) -> list[str]:
        colors: list[str] = [
            field.default
            for field in self.__dataclass_fields__.values()
            if isinstance(field.default, str)
        ]
        return colors

    @property
    def light(self) -> list[str]:
        return [self.green, self.blue, self.pink]

    @property
    def dark(self) -> list[str]:
        return [
            self.denim,
            self.navy,
            self.midnight,
            self.paleblue,
            self.palegreen,
            self.gray,
        ]

    @property
    def colorblind_paired(self) -> list[str]:
        return [
            self.bluish_green,
            self.redpurple,
            self.cb_blue,
            self.cb_orange,
            self.cb_yellow,
            self.cb_skyblue,
            self.cb_vermillion,
        ]

    @property
    def blues(self) -> list[str]:
        return [
            self.blue,
            self.ice,
            self.deepice,
            self.paleblue,
            self.denim,
            self.navy,
            self.midnight,
        ]

    @property
    def pinks(self) -> list[str]:
        return [self.pink, self.brightpink]

    @property
    def purples(self) -> list[str]:
        return [self.palepurple, self.purple]

    @property
    def neutral(self) -> list[str]:
        return [self.white, self.gray, self.midnight]

    @property
    def paired(self) -> list[str]:
        return [
            self.green,
            self.palegreen,
            self.pink,
            self.brightpink,
            self.palepurple,
            self.purple,
            self.ice,
            self.deepice,
            self.denim,
            self.navy,
            self.white,
            self.gray,
        ]

    @property
    def bright(self) -> list[str]:
        return [
            self.brightpink,
            self.teal,
            self.purple,
            self.denim,
            self.white,
            self.deepice,
        ]

    @property
    def pastel(self) -> list[str]:
        return [self.green, self.blue, self.pink, self.palepurple, self.gray]


class Theme:
    axis_linewidth: float = 2.8
    area_fill_color: str = "#ffb8b8"
    template: str = "presentation"
    axis_line_color: str = "black"
    axis_title_color: str = "black"
    tick_color: str = "#333"
    text_marker_line_color: str = "#333"
    marker_line_color: str = "#333"
    marker_pattern_bgcolor: str = "white"
    meanline_color: str = "#b3b3b3"
    box_fill_color: str = "#d6d6d6"


class LightTheme(Theme):
    template: str = "simple_white"
    tick_color: str = "#333"
    axis_line_color: str = "black"
    axis_title_color: str = "black"
    text_marker_line_color: str = "#333"
    box_fill_color: str = "rgba(214, 214, 214, 0.1)"
    marker_line_color: str = "#333"
    meanline_color: str = "#b3b3b3"
    violin_marker_line_color: str = "rgba(214, 214, 214, 0.1)"
    marker_pattern_bgcolor: str = "white"


class DarkTheme(Theme):
    template: str = "plotly_dark"
    tick_color: str = "#d6d6d6"
    axis_line_color: str = "white"
    axis_title_color: str = "white"
    marker_line_color: str = "#d6d6d6"
    text_marker_line_color: str = "#d6d6d6"
    box_fill_color: str = "#212121"
    meanline_color: str = "#616161"
    violin_marker_line_color: str = "rgba(33, 33, 33, 0.1)"
    marker_pattern_bgcolor: str = "black"


def format_axes(
    fig: px.Figure,
    x: str,
    y: str,
    theme: Theme = LightTheme(),
    labels: dict[str, str] = {},
    hide_xaxis_title: bool = False,
    hide_yaxis_title: bool = False,
    custom_xaxis_title: str | None = None,
    custom_yaxis_title: str | None = None,
    grid_x: bool = True,
    grid_y: bool = True,
    shared_xaxis_title: bool = False,
    shared_yaxis_title: bool = False,
    shared_xaxis_y_loc: float = -0.1,
    shared_yaxis_x_loc: float = -0.1,
    x_tick_labels: bool = True,
    y_tick_labels: bool = True,
) -> px.Figure:
    if custom_xaxis_title:
        x_title_text = custom_xaxis_title
    elif hide_xaxis_title:
        x_title_text = ""
    else:
        x_title_text = labels.get(x, x)

    if custom_yaxis_title:
        y_title_text = custom_yaxis_title
    elif hide_yaxis_title:
        y_title_text = ""
    else:
        y_title_text = labels.get(y, y)

    # if hide_xaxis_title:
    fig = futils.remove_xaxis_titles(fig)
    # if hide_yaxis_title:
    fig = futils.remove_yaxis_titles(fig)
    fig = futils.update_axes(
        fig,
        "y",
        theme.axis_line_color,
        theme.tick_color,
        theme.axis_title_color,
        showgrid=grid_y,
        linewidth=theme.axis_linewidth,
        title_text=y_title_text,
        showticklabels=y_tick_labels,
    )
    fig = futils.update_axes(
        fig,
        "x",
        theme.axis_line_color,
        theme.tick_color,
        theme.axis_title_color,
        showgrid=grid_x,
        linewidth=theme.axis_linewidth,
        title_text=x_title_text,
        showticklabels=x_tick_labels,
    )
    if shared_xaxis_title:
        x_title = "<b>" + x_title_text + "</b>"
        fig = futils.add_xaxis_title(fig, x_title, y=shared_xaxis_y_loc)
        fig = futils.remove_xaxis_titles(fig)

    if shared_yaxis_title:
        y_title = "<b>" + y_title_text + "</b>"
        fig = futils.add_yaxis_title(fig, y_title, x=shared_yaxis_x_loc)
        fig = futils.remove_yaxis_titles(fig)

    if shared_xaxis_title:
        margin_bottom = 75
    elif not hide_xaxis_title:
        margin_bottom = 25
    else:
        margin_bottom = 5

    fig = futils.update_layout(
        fig,
        font=dict(size=24, family="Helvetica"),
        margin=dict(
            l=120,
            r=50,
            b=margin_bottom,
            t=100,
        ),
    )
    return fig


def format_text(
    fig: px.Figure,
    text: str | None = None,
    theme: Theme = LightTheme(),
    text_template: str = "%{text:.1f}%",
) -> px.Figure:
    fig = futils.bold_trace_name(fig)
    fig = futils.bold_annotations(fig)
    if text:
        fig = futils.format_text_template(
            fig,
            marker_line_color=theme.text_marker_line_color,
            text_template=text_template,
        )
    return fig


def format_legend(
    fig: px.Figure,
    show_legend: bool = True,
    hide_legend_title: bool = False,
) -> px.Figure:
    if hide_legend_title:
        fig = futils.remove_legend_title(fig)
    fig = fig.update_layout(showlegend=show_legend)
    return fig


def format_facets(
    fig: px.Figure,
    data: pd.DataFrame,
    facet_col: str | None = None,
    facet_col_wrap: int | None = None,
    show_facet_y_line: bool = True,
) -> px.Figure:
    if not facet_col:
        return fig
    n_facets = len(set(data[facet_col]))
    multi_row = True
    if facet_col_wrap:
        multi_row = (n_facets / facet_col_wrap) > 1
    # hide shared y-axis tick labels when all in one row
    show_secondary_y_ticks = multi_row
    axis_keys = [f"yaxis{i+1}" for i in range(n_facets) if i > 0]
    secondary_axis = dict(
        showticklabels=show_secondary_y_ticks,
        showline=show_facet_y_line,
        tickfont=dict(color="rgba(0,0,0,0)"),
        title_text="",
        tickwidth=0.0,
        tickcolor="rgba(0,0,0,0)",
    )
    for axis_key in axis_keys:
        try:
            fig.layout[axis_key].update(secondary_axis)
        except KeyError:
            continue
    return fig


def apply_formatting(
    fig: px.Figure,
    data: pd.DataFrame,
    x: str,
    y: str,
    theme: Theme = LightTheme(),
    labels: dict[str, str] = {},
    hide_xaxis_title: bool = False,
    hide_yaxis_title: bool = False,
    custom_xaxis_title: str | None = None,
    custom_yaxis_title: str | None = None,
    grid_x: bool = True,
    grid_y: bool = True,
    shared_xaxis_title: bool = False,
    shared_yaxis_title: bool = False,
    shared_xaxis_y_loc: float = -0.1,
    shared_yaxis_x_loc: float = -0.1,
    x_tick_labels: bool = True,
    y_tick_labels: bool = True,
    facet_col: str | None = None,
    facet_col_wrap: int | None = None,
    show_facet_y_line: bool = True,
    show_legend: bool = True,
    hide_legend_title: bool = False,
    text: str | None = None,
    text_template: str = "%{text:.1f}%",
) -> px.Figure:
    fig = format_axes(
        fig,
        x=x,
        y=y,
        theme=theme,
        labels=labels,
        hide_xaxis_title=hide_xaxis_title,
        hide_yaxis_title=hide_yaxis_title,
        custom_xaxis_title=custom_xaxis_title,
        custom_yaxis_title=custom_yaxis_title,
        grid_x=grid_x,
        grid_y=grid_y,
        shared_xaxis_title=shared_xaxis_title,
        shared_yaxis_title=shared_yaxis_title,
        x_tick_labels=x_tick_labels,
        y_tick_labels=y_tick_labels,
        shared_xaxis_y_loc=shared_xaxis_y_loc,
        shared_yaxis_x_loc=shared_yaxis_x_loc,
    )
    fig = format_text(fig, text=text, theme=theme, text_template=text_template)
    fig = format_legend(
        fig, show_legend=show_legend, hide_legend_title=hide_legend_title
    )
    fig = format_facets(
        fig=fig,
        data=data,
        facet_col=facet_col,
        facet_col_wrap=facet_col_wrap,
        show_facet_y_line=show_facet_y_line,
    )
    return fig


class LinePlot:
    def __init__(self, theme: Theme = DarkTheme()) -> None:
        self.theme = theme

    def lineplot(
        self,
        data: pd.DataFrame,
        x: str,
        y: str,
        color: str | None = None,
        facet_col: str | None = None,
        facet_col_wrap: int | None = None,
        facet_row: str | None = None,
        text: str | None = None,
        color_discrete_map: dict[str, str] | None = None,
        labels: dict[str, str] = {},
        category_orders: dict[str, list[str]] | None = None,
        height: int = 650,
        width: int | None = None,
        hide_xaxis_title: bool = False,
        hide_yaxis_title: bool = False,
        custom_xaxis_title: str | None = None,
        custom_yaxis_title: str | None = None,
        grid_x: bool = True,
        grid_y: bool = True,
        show_legend: bool = True,
        hide_legend_title: bool = False,
        shared_xaxis_title: bool = False,
        shared_yaxis_title: bool = False,
        shared_xaxis_y_loc: float = -0.1,
        shared_yaxis_x_loc: float = -0.1,
        x_tick_labels: bool = True,
        y_tick_labels: bool = True,
        show_facet_y_line: bool = True,
        text_template: str = "%{text:.1f}%",
        line_dash: str | None = None,
    ) -> px.Figure:
        fig = px.line(
            data,
            x=x,
            y=y,
            color=color,
            facet_col=facet_col,
            color_discrete_map=color_discrete_map,
            height=height,
            width=width,
            labels=labels,
            category_orders=category_orders,
            template=self.theme.template,
            text=text,
            facet_row=facet_row,
            facet_col_wrap=facet_col_wrap,
            line_dash=line_dash,
        )
        fig = apply_formatting(
            fig=fig,
            data=data,
            x=x,
            y=y,
            theme=self.theme,
            labels=labels,
            hide_xaxis_title=hide_xaxis_title,
            hide_yaxis_title=hide_yaxis_title,
            custom_xaxis_title=custom_xaxis_title,
            custom_yaxis_title=custom_yaxis_title,
            grid_x=grid_x,
            grid_y=grid_y,
            shared_xaxis_title=shared_xaxis_title,
            shared_yaxis_title=shared_yaxis_title,
            shared_xaxis_y_loc=shared_xaxis_y_loc,
            shared_yaxis_x_loc=shared_yaxis_x_loc,
            x_tick_labels=x_tick_labels,
            y_tick_labels=y_tick_labels,
            facet_col=facet_col,
            facet_col_wrap=facet_col_wrap,
            show_facet_y_line=show_facet_y_line,
            show_legend=show_legend,
            hide_legend_title=hide_legend_title,
            text=text,
            text_template=text_template,
        )
        fig = fig.update_traces(line=dict(width=5))
        fig = fig.update_layout(
            legend=dict(orientation="h", yanchor="bottom", y=0.95, xanchor="right", x=1)
        )
        return fig


class BarPlot:
    def __init__(self, theme: Theme = DarkTheme()) -> None:
        self.theme = theme

    def barplot(
        self,
        data: pd.DataFrame,
        x: str,
        y: str,
        color: str | None = None,
        facet_col: str | None = None,
        facet_col_wrap: int | None = None,
        facet_row: str | None = None,
        text: str | None = None,
        color_discrete_map: dict[str, str] | None = None,
        labels: dict[str, str] = {},
        category_orders: dict[str, list[str]] | None = None,
        height: int = 650,
        width: int | None = None,
        barmode: str = "group",
        hide_xaxis_title: bool = False,
        hide_yaxis_title: bool = False,
        custom_xaxis_title: str | None = None,
        custom_yaxis_title: str | None = None,
        grid_x: bool = True,
        grid_y: bool = True,
        show_legend: bool = True,
        hide_legend_title: bool = False,
        shared_xaxis_title: bool = False,
        shared_yaxis_title: bool = False,
        show_facet_y_line: bool = True,
        text_template: str = "%{text:.1f}%",
        x_tick_labels: bool = True,
        y_tick_labels: bool = True,
        shared_xaxis_y_loc: float = -0.1,
        shared_yaxis_x_loc: float = -0.1,
    ) -> px.Figure:
        fig = px.bar(
            data,
            x=x,
            y=y,
            color=color,
            facet_col=facet_col,
            color_discrete_map=color_discrete_map,
            height=height,
            width=width,
            labels=labels,
            category_orders=category_orders,
            template=self.theme.template,
            text=text,
            facet_row=facet_row,
            facet_col_wrap=facet_col_wrap,
            barmode=barmode,
        )
        fig = apply_formatting(
            fig=fig,
            data=data,
            x=x,
            y=y,
            theme=self.theme,
            labels=labels,
            hide_xaxis_title=hide_xaxis_title,
            hide_yaxis_title=hide_yaxis_title,
            custom_xaxis_title=custom_xaxis_title,
            custom_yaxis_title=custom_yaxis_title,
            grid_x=grid_x,
            grid_y=grid_y,
            shared_xaxis_title=shared_xaxis_title,
            shared_yaxis_title=shared_yaxis_title,
            shared_xaxis_y_loc=shared_xaxis_y_loc,
            shared_yaxis_x_loc=shared_yaxis_x_loc,
            x_tick_labels=x_tick_labels,
            y_tick_labels=y_tick_labels,
            facet_col=facet_col,
            facet_col_wrap=facet_col_wrap,
            show_facet_y_line=show_facet_y_line,
            show_legend=show_legend,
            hide_legend_title=hide_legend_title,
            text=text,
            text_template=text_template,
        )
        return fig

    def grouped_stacked_bar(
        self,
        data: pd.DataFrame,
        x: str,
        group_col: str,
        y_cols: list[str],
        group_colors: dict[str, str] | None = {
            "Top 1": Colors.green,
            "Top 5": Colors.pink,
            "Oracle": Colors.blue,
        },
        x_order: list[str] | None = None,
        labels: dict[str, str] = {},
        pattern_cols: list[str] | None = ["percent_capri"],
        y_title: str | None = None,
        y_colors: dict[str, str] | None = {},
        width: int = 1400,
        height: int = 900,
        legend_orientation: str = "v",
        legend_font_size: int = 12,
        font_size: int = 16,
        font_family: str = "Helvetica",
        barmode: str = "group",
    ) -> px.Figure:
        df = data[[x, group_col] + y_cols].copy()
        df = df.sort_values(y_cols[0], ascending=False).reset_index(drop=True)
        df = df.pivot(index=x, values=y_cols, columns=group_col)

        # Set custom x-axis order
        if x_order:
            reorderlist = [m for m in x_order if m in df.index]
            df = df.iloc[pd.Categorical(df.index, reorderlist).argsort()].copy()

        groups = list(set(data[group_col]))
        n_groups = len(groups)
        if not group_colors:
            group_colors = {
                group: color
                for group, color in zip(groups, Colors().palette("paired", len(groups)))
            }
        # Preserve legend order to match group_colors (unordered if not provided)
        groups = list(group_colors.keys())

        color_map: dict[str, dict[str, str]] = {group: {} for group in groups}
        for group in groups:
            for y_col in y_cols:
                if y_colors:
                    y_color = y_colors[y_col]
                else:
                    y_color = group_colors[group]

                color_map[group][y_col] = y_color

        yaxes = {}
        for i in range(n_groups):
            if i == 0:
                continue
            axis_id = f"yaxis{i+1}"
            # Secondary y-axis overlayed on the primary one and not visible
            yaxes[axis_id] = go.layout.YAxis(
                visible=False,
                matches="y",
                overlaying="y",
                anchor="x",
            )
        fig = go.Figure(
            layout=go.Layout(
                height=height,
                width=width,
                template=self.theme.template,
                barmode=barmode,
                yaxis_showticklabels=True,
                yaxis_showgrid=True,
                yaxis_title=y_title or y_cols[0],
                yaxis_range=[0, min([df.values.max() * 1.4, 100])],
                font=dict(size=font_size, family=font_family),
                legend_orientation=legend_orientation,
                legend_font_size=legend_font_size,
                hovermode="x",
                margin=dict(b=10, t=10, l=10, r=10, pad=4),
                **yaxes,
            )
        )

        if not pattern_cols:
            pattern_cols = [y_cols[0]]

        # Add the traces
        for i, t in enumerate(y_cols):
            for j, col in enumerate(groups):
                if (df[t][col] == 0).all():
                    continue
                fig.add_bar(
                    x=df.index,
                    y=df[t][col],
                    # Set the right yaxis depending on the selected product (from enumerate)
                    yaxis=f"y{j + 1}",
                    # Offset the bar trace, offset needs to match the width
                    # For categorical traces, each category is spaced by 1
                    offsetgroup=j,
                    offset=(j - 1) * 1 / (n_groups + 1),
                    width=1 / (n_groups + 1),
                    legendgroup=col,
                    legendgrouptitle_text=col,
                    name=labels.get(t, t),
                    marker_color=color_map[col][t],
                    # text=sr_holo[t][col].round(1),
                    marker_line=dict(width=2, color=self.theme.marker_line_color),
                    marker_pattern_shape="\\" if t in pattern_cols else "",
                    marker_pattern_bgcolor=self.theme.marker_pattern_bgcolor
                    if t in pattern_cols
                    else None,
                    marker_pattern_fgcolor=color_map[col][t]
                    if t in pattern_cols
                    else None,
                    marker_pattern_fgopacity=1.0,
                    marker_pattern_solidity=0.4 if t in pattern_cols else None,
                    hovertemplate="%{y}<extra></extra>",
                )
        fig = fig.update_layout(
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.01)
        )
        return fig


class ViolinPlot:
    def __init__(self, theme: Theme = DarkTheme()) -> None:
        self.theme = theme

    def violinplot(
        self,
        data: pd.DataFrame,
        x: str,
        y: str,
        color: str | None = None,
        box: bool = True,
        facet_col: str | None = None,
        facet_col_wrap: int | None = None,
        facet_col_spacing: float | None = None,
        text: str | None = None,
        points: str = "outliers",
        color_discrete_map: dict[str, str] | None = None,
        labels: dict[str, str] = {},
        category_orders: dict[str, list[str]] | None = None,
        height: int = 650,
        width: int | None = None,
        violinmode: str = "group",
        hide_xaxis_title: bool = False,
        hide_yaxis_title: bool = False,
        custom_xaxis_title: str | None = None,
        custom_yaxis_title: str | None = None,
        grid_x: bool = True,
        grid_y: bool = True,
        show_legend: bool = True,
        hide_legend_title: bool = False,
        shared_xaxis_title: bool = False,
        shared_yaxis_title: bool = False,
        shared_xaxis_y_loc: float = -0.1,
        shared_yaxis_x_loc: float = -0.1,
        hard_span: bool = False,
        span: list[float] | None = None,
        vrect: tuple[float, float] | None = None,
        hrect: tuple[float, float] | None = None,
        meanline_visible: bool = True,
        scalemode: str = "width",
        scale_width: float = 0.6,
        x_tick_labels: bool = True,
        y_tick_labels: bool = True,
        marker_line_width: float = 0.5,
        marker_size: int = 8,
        marker_opacity: float = 0.9,
        show_facet_y_line: bool = True,
        text_template: str = "%{text:.1f}%",
    ) -> px.Figure:
        fig = px.violin(
            data,
            template=self.theme.template,
            x=x,
            y=y,
            color=color,
            color_discrete_map=color_discrete_map,
            points=points,
            box=box,
            category_orders=category_orders,
            height=height,
            violinmode=violinmode,
            facet_col=facet_col,
            facet_col_wrap=facet_col_wrap,
            facet_col_spacing=facet_col_spacing,
            width=width,
        )
        if hard_span:
            fig.update_traces(spanmode="hard")
        if span:
            fig.update_traces(span=span, spanmode="manual")

        fig = apply_formatting(
            fig=fig,
            data=data,
            x=x,
            y=y,
            theme=self.theme,
            labels=labels,
            hide_xaxis_title=hide_xaxis_title,
            hide_yaxis_title=hide_yaxis_title,
            custom_xaxis_title=custom_xaxis_title,
            custom_yaxis_title=custom_yaxis_title,
            grid_x=grid_x,
            grid_y=grid_y,
            shared_xaxis_title=shared_xaxis_title,
            shared_yaxis_title=shared_yaxis_title,
            shared_xaxis_y_loc=shared_xaxis_y_loc,
            shared_yaxis_x_loc=shared_yaxis_x_loc,
            x_tick_labels=x_tick_labels,
            y_tick_labels=y_tick_labels,
            facet_col=facet_col,
            facet_col_wrap=facet_col_wrap,
            show_facet_y_line=show_facet_y_line,
            show_legend=show_legend,
            hide_legend_title=hide_legend_title,
            text=text,
            text_template=text_template,
        )
        fig = futils.update_layout(
            fig,
            font=dict(size=24, family="Helvetica"),
            margin=dict(
                l=140,
                r=50,
                b=100 if facet_col else 0,
                t=80,
            ),
        )
        fig.update_traces(
            marker_line=dict(
                width=marker_line_width, color=self.theme.marker_line_color
            ),
            meanline_visible=meanline_visible,
            marker_size=marker_size,
            marker_opacity=marker_opacity,
        )
        fig.for_each_trace(
            lambda t: t.update(
                width=scale_width,
                scalemode=scalemode,
                opacity=1.0,
                meanline=dict(
                    visible=meanline_visible, color=self.theme.meanline_color, width=2
                ),
                box=dict(
                    visible=box,
                    width=0.2,
                    fillcolor=self.theme.box_fill_color,
                    line=dict(width=1, color=self.theme.marker_line_color),
                ),
            )
        )
        if vrect:
            fig.add_vrect(
                x0=vrect[0],
                x1=vrect[1],
                fillcolor=self.theme.area_fill_color,
                opacity=0.1,
                line_width=1,
            )
        if hrect:
            fig.add_hrect(
                y0=hrect[0],
                y1=hrect[1],
                fillcolor=self.theme.area_fill_color,
                opacity=0.1,
                line_width=1,
            )
        return fig
