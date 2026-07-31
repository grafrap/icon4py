# Standalone figure generator for the V2E2C2V composition (RbfNabla4 vertex stencil).
# Mirrors the color conventions of model/atmosphere/dycore/docs/ext/offset_providers.py:
#   blue = cell, orange = edge, green = vertex.
#
# Ring vertex positions are taken from the real V2E2C2V connectivity table in
# gt4py's map_dict.py (../gt4py/src/gt4py/next/iterator/transforms/map_dict.py),
# slot -> (dI, dJ):
#   0=(0,0) center=v0, 1=(0,1), 2=(-1,1), 3=(1,0), 4=(1,-1), 5=(0,-1), 6=(-1,0)
# converted to angles using the project's mesh basis (dJ=+1 -> East/0 deg,
# dI=+1 -> Northeast/60 deg), which places the slots counterclockwise as
# 6(0 deg) -> 2(60 deg) -> 1(120 deg) -> 3(180 deg) -> 4(240 deg) -> 5(300 deg).
#
# Left panel: the two-stage decomposition actually used by rbf_nabla4.py
#   (CalculateNabla4 = E2C2V, then RbfVecInterpolVertex = V2E).
# Right panel: the fused V2E2C2V connectivity as it is directly represented
#   in map_dict.py -- a single 7-slot (1 self + 6 neighbor) vertex-to-vertex
#   table, with no edge/cell intermediate.
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np


SIDE = 1.0
COLORS = list(mcolors.TABLEAU_COLORS.values())
BLUE, ORANGE, GREEN, PURPLE = COLORS[0], COLORS[1], COLORS[2], COLORS[4]

# ring slot -> angle (degrees), counterclockwise, derived from the map_dict (dI,dJ) offsets
SLOT_ANGLE_DEG = {6: 0, 2: 60, 1: 120, 3: 180, 4: 240, 5: 300}
RING_ORDER = [6, 2, 1, 3, 4, 5]  # counterclockwise, starting at angle 0


def slot_vertex(slot: int) -> tuple[float, float]:
    theta = np.deg2rad(SLOT_ANGLE_DEG[slot])
    return (SIDE * np.cos(theta), SIDE * np.sin(theta))


def draw_mesh_outline(ax, v0, ring):
    for i in range(6):
        tri = plt.Polygon(
            [v0, ring[i], ring[(i + 1) % 6]], edgecolor="black", fill=None, linewidth=1.0, zorder=1
        )
        ax.add_patch(tri)


def draw_vertices(ax, v0, ring):
    for p in ring:
        ax.plot(p[0], p[1], "o", color=GREEN, markersize=11, zorder=5, markeredgecolor="black", markeredgewidth=0.6)
    ax.plot(v0[0], v0[1], "o", color=GREEN, markersize=17, zorder=6, markeredgecolor="black", markeredgewidth=1.3)


def draw_two_stage_panel(ax, v0, ring, emph):
    # highlight the two cells sharing the emphasized edge
    for i in ((emph - 1) % 6, emph):
        tri = plt.Polygon(
            [v0, ring[i], ring[(i + 1) % 6]], facecolor=BLUE, alpha=0.25, edgecolor=None, zorder=0
        )
        ax.add_patch(tri)

    # --- Stage 2: V2E -- gather all 6 incident edges into v0 (orange, arrows point into v0)
    # Only drawn from the edge midpoint to v0 (not from the far ring vertex) so this
    # cannot be misread as a direct vertex-to-vertex (V2V) connection.
    for i in range(6):
        mid = ((v0[0] + ring[i][0]) / 2, (v0[1] + ring[i][1]) / 2)
        ax.annotate(
            "", xy=v0, xytext=mid,
            arrowprops=dict(facecolor=ORANGE, edgecolor=ORANGE, shrink=0.0, width=1.1, headwidth=8),
            zorder=3,
        )

    # --- Stage 1: E2C2V -- for the emphasized edge, gather its 4 diamond vertices onto the edge
    apex_left = ring[(emph - 1) % 6]
    apex_right = ring[(emph + 1) % 6]
    mid_emph = ((v0[0] + ring[emph][0]) / 2, (v0[1] + ring[emph][1]) / 2)
    for src in (v0, ring[emph], apex_left, apex_right):
        ax.annotate(
            "", xy=mid_emph, xytext=src,
            arrowprops=dict(facecolor=PURPLE, edgecolor=PURPLE, shrink=0.12, width=1.0, headwidth=7),
            zorder=4,
        )

    draw_vertices(ax, v0, ring)
    ax.text(mid_emph[0], mid_emph[1] + 0.16, "$e^*$", fontsize=12, ha="center", va="center",
            color=ORANGE, fontweight="bold")

    legend_elems = [
        plt.Line2D([0], [0], color=PURPLE, lw=1.5, label=r"E2C2V: 4 diamond vertices $\to$ edge (CalculateNabla4)"),
        plt.Line2D([0], [0], color=ORANGE, lw=2.2, label=r"V2E: edge $\to$ central vertex (RbfVecInterpolVertex)"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=GREEN, markeredgecolor="black",
                    markersize=9, label=r"vertex field ($u_{vert}, v_{vert}$)"),
        mpatches.Patch(facecolor=BLUE, alpha=0.25, label=r"cell pair sharing edge $e^*$"),
    ]
    ax.legend(handles=legend_elems, loc="upper center", bbox_to_anchor=(0.5, 0.04),
              ncol=1, fontsize=8.5, frameon=False)
    ax.set_title("Two-stage neighbor connectivity: E2C2V $\\to$ V2E", fontsize=11, pad=4)



def draw_fused_panel(ax, v0, ring):
    # Direct vertex-to-vertex arrows: the fused V2E2C2V offset in map_dict.py reads
    # each of the 6 ring vertices straight into v0, with no edge/cell intermediate.
    for p in ring:
        ax.annotate(
            "", xy=v0, xytext=p,
            arrowprops=dict(facecolor=GREEN, edgecolor=GREEN, shrink=0.12, width=1.1, headwidth=8),
            zorder=3,
        )

    draw_vertices(ax, v0, ring)

    legend_elems = [
        plt.Line2D([0], [0], color=GREEN, lw=2.0, label=r"V2E2C2V (fused): vertex $\to$ central vertex, directly"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=GREEN, markeredgecolor="black",
                    markersize=9, label=r"vertex field ($u_{vert}, v_{vert}$)"),
    ]
    ax.legend(handles=legend_elems, loc="upper center", bbox_to_anchor=(0.5, 0.04),
              ncol=1, fontsize=8.5, frameon=False)
    ax.set_title("V2E2C2V neighbor connectivity", fontsize=11, pad=4)


def main():
    v0 = (0.0, 0.0)
    ring = [slot_vertex(s) for s in RING_ORDER]
    emph_slot = 2
    emph = RING_ORDER.index(emph_slot)  # index into `ring` of the emphasized edge

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 5.6))

    for ax in (ax1, ax2):
        draw_mesh_outline(ax, v0, ring)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.05, 1.05)

    draw_two_stage_panel(ax1, v0, ring, emph)
    draw_fused_panel(ax2, v0, ring)

    fig.suptitle("V2E2C2V composition in RBF Nabla4", fontsize=14, y=0.96)
    fig.subplots_adjust(top=0.86, bottom=0.16, wspace=0.05)

    fig.savefig("v2e2c2v_figure.png", dpi=300, bbox_inches="tight")
    fig.savefig("v2e2c2v_figure.pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()
