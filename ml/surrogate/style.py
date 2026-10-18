"""Plot style shared by the notebooks.

    from surrogate import style
    style.apply()                                   # once, at the top of a notebook
    ax.plot(f, y, color=style.MESH_COLORS["m45x15"])
    ax.scatter(x, y, c=values, cmap=style.SEQ)

Rules kept throughout: a footprint keeps its colour in every figure (categorical slots in a
fixed, colour-blind-checked order); a continuous input (H, g, f0) uses one sequential hue,
light to dark; grids and axes stay quiet so the data carry the contrast.
"""
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

# Categorical slots, in a validated order (adjacent pairs separable under colour-vision deficiency).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
MESH_ORDER = ["m30x15", "m45x15", "m60x15", "m80x20", "ref33x31"]
MESH_COLORS = dict(zip(MESH_ORDER, SERIES))

# Sequential ramp for continuous inputs: one hue (blue), light -> dark; the lightest step stays
# visible on a white background.
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])

INK = "#0b0b0b"        # titles, labels
INK2 = "#52514e"       # ticks, secondary text
GRID = "#e4e3df"       # grid lines


def apply():
    """Matplotlib defaults for all notebook figures."""
    mpl.rcParams.update({
        "figure.dpi": 110,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": INK2,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.prop_cycle": mpl.cycler(color=SERIES),
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "font.size": 9,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "lines.linewidth": 2,
    })
