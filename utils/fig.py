import os
from typing import (Any, Callable, Iterator, List, Literal, Optional, Sequence,
                    Tuple, overload)

import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
import numpy as np
import seaborn as sns
from matplotlib.axis import Axis
from matplotlib.cm import ScalarMappable
from matplotlib.collections import PolyCollection
from matplotlib.colorbar import Colorbar
from matplotlib.contour import QuadContourSet
from matplotlib.pyplot import Axes as PltAxes
from matplotlib.quiver import Quiver


class Axes:
    def __init__(self, ax: PltAxes) -> None:
        self.ax = ax

    def remove(self) -> None:
        self.ax.remove()

    def remove_grid(self) -> None:
        self.ax.grid(False)

    def set_spines_linewidth(self, w: float) -> None:
        for axis in ['top','bottom','left','right']:
            self.ax.spines[axis].set_linewidth(w)

    def set_title(self, title: str, fontsize: Optional[float] = None) -> None:
        fontsize = plt.rcParams['axes.titlesize'] if fontsize is None else fontsize
        self.ax.set_title(title, fontsize=fontsize)

    def set_xlabel(self, label: str, fontsize: Optional[float] = None, labelpad: Optional[float] = None) -> None:
        fontsize = plt.rcParams['axes.labelsize'] if fontsize is None else fontsize
        self.ax.set_xlabel(label, fontsize=fontsize, labelpad=labelpad) # type: ignore

    def set_ylabel(self, label: str, fontsize: Optional[float] = None, labelpad: Optional[float] = None) -> None:
        fontsize = plt.rcParams['axes.labelsize'] if fontsize is None else fontsize
        self.ax.set_ylabel(label, fontsize=fontsize, labelpad=labelpad) # type: ignore

    def set_xlim(self, left: float, right: float) -> None:
        self.ax.set_xlim(left, right)

    def set_ylim(self, bottom: float, top: float) -> None:
        self.ax.set_ylim(bottom, top)

    def set_xticks(self, ticks: Sequence[float]) -> None:
        self.ax.set_xticks(ticks)

    def set_yticks(self, ticks: Sequence[float]) -> None:
        self.ax.set_yticks(ticks)

    def set_xticklabels(self, ticks: Sequence[str]) -> None:
        self.ax.set_xticklabels(ticks)

    def set_yticklabels(self, ticks: Sequence[str]) -> None:
        self.ax.set_yticklabels(ticks)

    def set_fontsize(self, fontsize: float) -> None:
        self.set_title_fontsize(fontsize)
        self.set_label_fontsize(fontsize)
        self.set_ticks_fontsize(fontsize)
        self.set_legend_fontsize(fontsize)

    def set_title_fontsize(self, fontsize: float) -> None:
        self.ax.set_title(self.ax.get_title(), fontsize=fontsize)

    def set_label_fontsize(self, fontsize: float) -> None:
        self.set_xlabel_fontsize(fontsize)
        self.set_ylabel_fontsize(fontsize)

    def set_xlabel_fontsize(self, fontsize: float) -> None:
        self.ax.set_xlabel(self.ax.get_xlabel(), fontsize=fontsize)

    def set_ylabel_fontsize(self, fontsize: float) -> None:
        self.ax.set_ylabel(self.ax.get_ylabel(), fontsize=fontsize)

    def set_ticks_fontsize(self, fontsize: float) -> None:
        self.ax.tick_params(axis='both', which='major', labelsize=fontsize)

    def set_xticks_fontsize(self, fontsize: float) -> None:
        self.ax.tick_params(axis='x', which='major', labelsize=fontsize)

    def set_yticks_fontsize(self, fontsize: float) -> None:
        self.ax.tick_params(axis='y', which='major', labelsize=fontsize)

    def remove_xticks(self) -> None:
        self.ax.tick_params(bottom=False, labelbottom=False)

    def remove_yticks(self) -> None:
        self.ax.tick_params(left=False, labelleft=False)

    def remove_ticks(self) -> None:
        self.remove_xticks()
        self.remove_yticks()

    def set_legend_fontsize(self, fontsize: float) -> None:
        self.ax.legend(fontsize=fontsize)

    def set_xticks_offset(self, use: bool) -> None:
        self.ax.xaxis.set_major_formatter(tkr.ScalarFormatter(use))

    def set_yticks_offset(self, use: bool) -> None:
        self.ax.yaxis.set_major_formatter(tkr.ScalarFormatter(use))

    def set_xticks_comma(self) -> None:
        self.ax.xaxis.set_major_formatter(tkr.FuncFormatter(lambda x, _: f'{int(x):,}'))

    def set_yticks_comma(self) -> None:
        self.ax.yaxis.set_major_formatter(tkr.FuncFormatter(lambda x, _: f'{int(x):,}'))

    @staticmethod
    def _move_label(axis: Axis, left: float, down: float) -> None:
        label = axis.get_label()
        x, y = label.get_position()
        label.set_position((x-left, y-down))

    def move_xlabel(self, left: float, down: float) -> None:
        self._move_label(self.ax.xaxis, left, down)

    def move_ylabel(self, left: float, down: float) -> None:
        self._move_label(self.ax.yaxis, left, down)

    def move_legend(
        self,
        loc: Literal['best', 'upper left'],
        bbox_to_anchor: Sequence[float] = (0, 0),
        ncols: int = 1,
        frameon: bool = True,
        borderpad: float = 0.4,
        columnspacing: float = 2.0,
    ) -> None:
        # borderpad: the fractional whitespace inside the legend border
        self.ax.legend(
            loc=loc,
            bbox_to_anchor=bbox_to_anchor, 
            ncols=ncols,
            frameon=frameon, 
            borderpad=borderpad,
            columnspacing=columnspacing, 
        )

    def remove_legend(self) -> None:
        self.ax.get_legend().remove() # type: ignore

    def set_clabel(
        self, 
        cs: QuadContourSet, 
        fontsize: Optional[float] = None, 
        inline: bool = True, 
        inline_spacing: float = 5, 
        fmt: Optional[Callable] = None,
        manual: Optional[Tuple[Tuple[float, float], ...]] = None,
    ) -> None:
        fontsize = plt.rcParams['xtick.labelsize'] if fontsize is None else fontsize
        self.ax.clabel(
            cs, 
            fontsize=fontsize,
            inline=inline, 
            inline_spacing=inline_spacing,
            fmt=fmt, 
            manual=manual,
        )

    def line(
        self,
        x: Any, 
        y: Any, 
        alpha: Optional[float] = None, 
        color: Any = None, 
        label: Optional[str] = None, 
        linestyle: str = '-', 
        linewidth: float = 3,
        marker: Optional[str] = None,
    ) -> PltAxes:
        return sns.lineplot(
            x=x, 
            y=y, 
            ax=self.ax,
            alpha=alpha, 
            color=color, 
            label=label, 
            linestyle=linestyle, 
            linewidth=linewidth,
            marker=marker, 
        )

    def line_with_band(
        self,
        x: Any, 
        center: Any, 
        diff: Any, 
        color: Any = None,
        label: Optional[str] = None,
        linewidth: float = 2,
        marker: Optional[str] = None,
    ) -> Tuple[PltAxes, PolyCollection]:
        if isinstance(center, (list, tuple)):
            center = np.array(center)
        if isinstance(diff, (list, tuple)):
            diff = np.array(diff)
        line = self.line(
            x, 
            center, 
            color=color, 
            label=label, 
            linewidth=linewidth, 
            marker=marker,
        )
        polycollection = self.ax.fill_between(
            x, 
            center - diff, 
            center + diff, 
            alpha=0.2, 
            color=color,
        )
        return line, polycollection

    def hist(
        self, 
        data: Any, 
        color: Any = None, 
        label: Optional[str] = None, 
        stat: str = 'density',
    ) -> PltAxes:
        return sns.histplot(
            data=data, 
            ax=self.ax, 
            color=color, 
            element='step', 
            label=label, 
            stat=stat,
        )

    def contour(
        self,
        x: Any,
        y: Any,
        z: Any,
        colors: Any = None,
        levels: Any = None,
        linewidth: float = 2,
        linestyles: str = 'solid',
    ) -> QuadContourSet:
        return self.ax.contour(
            x, y, z, 
            levels=levels, 
            colors=colors, 
            linewidths=[linewidth], 
            linestyles=linestyles,
        )

    def contourf(
        self,
        x: Any,
        y: Any,
        z: Any,
        cmap: Any = None,
        levels: Any = None,
    ) -> QuadContourSet:
        return self.ax.contourf(x, y, z, levels=levels, cmap=cmap)

    def imshow(
        self, 
        img: Any, 
        gray: bool = False, 
        vmin: float = 0,
        vmax: float = 1,
        remove_xticks: bool = True, 
        remove_yticks: bool = True
    ) -> None:
        if remove_xticks:
            self.remove_xticks()
        if remove_yticks:
            self.remove_yticks()
        l = len(img.shape)
        if l == 2:
            self.ax.imshow(img, cmap='gray', vmin=vmin, vmax=vmax)
        elif l == 3:
            if gray:
                for i in img:
                    self.ax.imshow(i, cmap='gray', vmin=vmin, vmax=vmax)
            else:
                self.ax.imshow(img, vmin=vmin, vmax=vmax)
        elif l == 4:
            for i in img:
                self.ax.imshow(i, vmin=vmin, vmax=vmax)
        else:
            raise ValueError(l)
        
    def horizontal_quiver(self, limit: float) -> Quiver:
        return self.ax.quiver(-limit, 0, limit, 0, angles='xy', scale_units='xy', scale=0.5)

    def vertical_quiver(self, limit: float) -> Quiver:
        return self.ax.quiver(0, -limit, 0, limit, angles='xy', scale_units='xy', scale=0.5)


class Figure:
    palette: Tuple[Tuple[float, float, float], ...] = sns.color_palette('deep') # type: ignore
    dark_palette: Tuple[Tuple[float, float, float], ...] = sns.color_palette('dark') # type: ignore
    light_palette: Tuple[Tuple[float, float, float], ...] = sns.color_palette('pastel') # type: ignore

    @staticmethod
    def set_seaborn_theme() -> None:
        sns.set_theme()

    @staticmethod
    def set_font_scale(font_scale: float) -> None:
        '''
        From `seaborn/rcmod.py`,
            ```
            texts_base_context = {
                "font.size": 12,
                "axes.labelsize": 12,
                "axes.titlesize": 12,
                "xtick.labelsize": 11,
                "ytick.labelsize": 11,
                "legend.fontsize": 11,
                "legend.title_fontsize": 12,
            }
            ```
        '''
        sns.set_context('notebook', font_scale) # type: ignore
        plt.rcParams['figure.labelsize'] = 12 * font_scale
        plt.rcParams['figure.titlesize'] = 12 * font_scale

    @classmethod
    def set_tex(cls, luatex: bool = False) -> None:
        plt.rcParams['text.usetex'] = True
        plt.rcParams['text.latex.preamble'] = r'''
            \usepackage{amsmath}
            \usepackage{bm}
        '''
        cls.set_font('tex')
        if luatex:
            # make Japanese label and TeX compatible
            cls.set_backend('lualatex')
            plt.rcParams['pgf.texsystem'] = 'lualatex'

    @classmethod
    def unset_tex(cls, luatex: bool = False) -> None:
        plt.rcParams['text.usetex'] = False
        if luatex:
            cls.set_backend('notebook')

    @staticmethod
    def set_backend(mode: Literal['notebook', 'lualatex']) -> None:
        if mode == 'notebook':
            backend = 'module://matplotlib_inline.backend_inline'
        elif mode == 'lualatex':
            backend = 'pgf'
        plt.rcParams['backend'] = backend

    @staticmethod
    def set_font(mode: Literal['default', 'tex', 'japanese']) -> None:
        if mode == 'default':
            plt.rcParams['font.family'] = ['sans-serif']
        elif mode == 'tex':
            plt.rcParams['font.family'] = 'cm'
        elif mode == 'japanese':
            # This is nearly equal to `plt.rcParams['font.family'] = 'IPAexGothic'`
            import japanize_matplotlib

    @staticmethod
    def set_mathfont(mode: Literal['default', 'tex']) -> None:
        # This is effective when `plt.rcParams['text.usetex'] = False`
        if mode == 'default':
            font = 'dejavusans'
        elif mode == 'tex':
            font = 'cm'
        plt.rcParams['mathtext.fontset'] = font

    @staticmethod
    def set_high_dpi() -> None:
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['savefig.dpi'] = 300

    @staticmethod
    def save(path: str, *paths: str) -> None:
        # For flexibly, do not use `self.fig.savefig(...)`
        p = os.path.join(path, *paths)
        plt.savefig(p, dpi=300, bbox_inches='tight', pad_inches=0.025)

    @staticmethod
    def show() -> None:
        plt.show()

    @staticmethod
    def close() -> None:
        plt.close()

    def __init__(
        self,
        n_row: int = 1,
        n_col: int = 1,
        figsize: Sequence[float] = (6.4, 4.8), 
    ) -> None:
        self.n_row = n_row
        self.n_col = n_col
        self.fig, self._axes = plt.subplots(n_row, n_col, figsize=figsize, layout='constrained')
        self.axes = self._convert_axes()

    def _convert_axes(self) -> List[List[Axes]]:
        if self.n_row == 1 and self.n_col == 1:
            return [[Axes(self._axes)]] # type: ignore
        elif self.n_row == 1:
            return [[Axes(ax) for ax in self._axes]] # type: ignore
        elif self.n_col == 1:
            return [[Axes(ax)] for ax in self._axes] # type: ignore
        else:
            return [[Axes(ax) for ax in row] for row in self._axes]
        
    def generate(self) -> Iterator[Axes]:
        # 0 1 2 3 4
        # 5 6 7 8 ...
        for row in range(self.n_row):
            for col in range(self.n_col):
                yield self.axes[row][col]

    def set_suptitle(self, label: str, fontsize: Optional[float] = None) -> None:
        fontsize = plt.rcParams['figure.labelsize'] if fontsize is None else fontsize
        self.fig.suptitle(label, fontsize=fontsize)

    def set_supxlabel(self, label: str, fontsize: Optional[float] = None) -> None:
        fontsize = plt.rcParams['figure.labelsize'] if fontsize is None else fontsize
        self.fig.supxlabel(label, fontsize=fontsize)

    def set_supylabel(self, label: str, fontsize: Optional[float] = None) -> None:
        fontsize = plt.rcParams['figure.labelsize'] if fontsize is None else fontsize
        self.fig.supylabel(label, fontsize=fontsize)

    @overload
    def set_axes_space(self, w_pad: float, h_pad: None) -> None:
        ...

    @overload
    def set_axes_space(self, w_pad: None, h_pad: float) -> None:
        ...

    @overload
    def set_axes_space(self, w_pad: float, h_pad: float) -> None:
        ...

    def set_axes_space(self, w_pad=None, h_pad=None) -> None:
        self.fig.set_constrained_layout_pads(w_pad=w_pad, h_pad=h_pad)

    def set_colorbar(
        self,
        mappable: ScalarMappable, 
        tick_fontsize: Optional[float] = None, 
        label: Optional[str] = None, 
        label_fontsize: Optional[float] = None,
        labelpad: Optional[int] = None,
        pad: float = 0.01,
    ) -> Colorbar:
        cbar = self.fig.colorbar(mappable, ax=self._axes, pad=pad)
        if tick_fontsize is not None:
            cbar.ax.tick_params(labelsize=tick_fontsize)
        if label is not None:
            if label_fontsize is None:
                label_fontsize = plt.rcParams['axes.labelsize']
            cbar.set_label(label, labelpad=labelpad, rotation=270, size=label_fontsize)
        return cbar