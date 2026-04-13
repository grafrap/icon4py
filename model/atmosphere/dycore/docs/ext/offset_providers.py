# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import os
from typing import Final

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np


SIDE: Final = 1  # Length of each triangle side
LABEL_TRIANGLES: Final = False  # Option to label each triangle center with its ID
COLORS: Final = list(mcolors.TABLEAU_COLORS.values())
AX_BORDER: Final = 0.1 * SIDE  # Border around the axes
IMG_DIR = "img"


class Triangle:
    def __init__(self, ax, x0, y0, orientation):
        self.ax = ax
        self.x0 = x0
        self.y0 = y0
        self.orientation = orientation
        self.calculate_vertices()
        self.label_offset = 0.10 * SIDE
        self.cell_offset = 0.25 * SIDE
        self.edge_offset = 0.25 * SIDE
        self.vertex_size = 10
        self.bold_line = 6
        self.edge_linewidth = 1.0

    def calculate_vertices(self):
        if self.orientation == "up":
            # Upward-pointing triangle
            self.A = (self.x0, self.y0)
            self.B = (self.x0 + SIDE, self.y0)
            self.C = (self.x0 + SIDE / 2, self.y0 + SIDE * np.sqrt(3) / 2)
        else:
            # Downward-pointing triangle
            self.A = (self.x0, self.y0)
            self.B = (self.x0 + SIDE / 2, self.y0 + SIDE * np.sqrt(3) / 2)
            self.C = (self.x0 - SIDE / 2, self.y0 + SIDE * np.sqrt(3) / 2)

        # Calculate the center
        self.CC = ((self.A[0] + self.B[0] + self.C[0]) / 3, (self.A[1] + self.B[1] + self.C[1]) / 3)

        # Calculate the midpoints of the edges
        self.AB = ((self.A[0] + self.B[0]) / 2, (self.A[1] + self.B[1]) / 2)
        self.BC = ((self.B[0] + self.C[0]) / 2, (self.B[1] + self.C[1]) / 2)
        self.CA = ((self.C[0] + self.A[0]) / 2, (self.C[1] + self.A[1]) / 2)

    def draw(self):
        # Draw the triangle edges
        triangle = plt.Polygon([self.A, self.B, self.C], edgecolor="black", fill=None, linewidth=self.edge_linewidth)
        self.ax.add_patch(triangle)

    def print_labels(self, tri_id):
        """
        Labels the triangle with the given label.
        Parameters:
            label : label identifying the triangle
        """
        self.ax.text(self.CC[0], self.CC[1], f"T{tri_id}", fontsize=8, ha="center")
        # add labels to vertices depending on orientation
        if self.orientation == "up":
            self.ax.text(
                self.A[0] + self.label_offset * np.cos(np.pi / 6),
                self.A[1] + self.label_offset * np.sin(np.pi / 6),
                "A",
                fontsize=8,
                ha="center",
                va="center",
            )
            self.ax.text(
                self.B[0] - self.label_offset * np.cos(np.pi / 6),
                self.B[1] + self.label_offset * np.sin(np.pi / 6),
                "B",
                fontsize=8,
                ha="center",
                va="center",
            )
            self.ax.text(
                self.C[0], self.C[1] - self.label_offset, "C", fontsize=8, ha="center", va="center"
            )
        else:
            self.ax.text(
                self.A[0], self.A[1] + self.label_offset, "A", fontsize=8, ha="center", va="center"
            )
            self.ax.text(
                self.B[0] - self.label_offset * np.cos(np.pi / 6),
                self.B[1] - self.label_offset * np.sin(np.pi / 6),
                "B",
                fontsize=8,
                ha="center",
                va="center",
            )
            self.ax.text(
                self.C[0] + self.label_offset * np.cos(np.pi / 6),
                self.C[1] - self.label_offset * np.sin(np.pi / 6),
                "C",
                fontsize=8,
                ha="center",
                va="center",
            )

    def color_vertex(self, vertex, coloridx=0):
        """
        Colors the vertex of the triangle.
        Parameters:
            vertex : Vertex to be colored ('A', 'B', or 'C')
        """
        vertex_size = self.vertex_size - 8# * coloridx
        match vertex:
            case "A":
                self.ax.plot(
                    self.A[0], self.A[1], "o", color=COLORS[coloridx], markersize=vertex_size
                )
            case "B":
                self.ax.plot(
                    self.B[0], self.B[1], "o", color=COLORS[coloridx], markersize=vertex_size
                )
            case "C":
                self.ax.plot(
                    self.C[0], self.C[1], "o", color=COLORS[coloridx], markersize=vertex_size
                )

    def color_vertices(self, coloridx=0):
        """
        Colors the vertices of the triangle.
        """
        self.color_vertex("A", coloridx)
        self.color_vertex("B", coloridx)
        self.color_vertex("C", coloridx)

    def color_cell(self, coloridx=0):
        """
        Fills the triangle area with color, leaving some distance from the sides.
        """
        cell_offset = self.cell_offset #+ self.cell_offset / 8 * coloridx
        if self.orientation == "up":
            A = (
                self.A[0] + cell_offset * np.cos(np.pi / 6),
                self.A[1] + cell_offset * np.sin(np.pi / 6),
            )
            B = (
                self.B[0] - cell_offset * np.cos(np.pi / 6),
                self.B[1] + cell_offset * np.sin(np.pi / 6),
            )
            C = (self.C[0], self.C[1] - cell_offset)
        else:
            A = (self.A[0], self.A[1] + cell_offset)
            B = (
                self.B[0] - cell_offset * np.cos(np.pi / 6),
                self.B[1] - cell_offset * np.sin(np.pi / 6),
            )
            C = (
                self.C[0] + cell_offset * np.cos(np.pi / 6),
                self.C[1] - cell_offset * np.sin(np.pi / 6),
            )

        # Draw the filled triangle
        filled_triangle = plt.Polygon([A, B, C], edgecolor=None, facecolor=COLORS[coloridx])
        self.ax.add_patch(filled_triangle)

    def color_edge(self, edge, coloridx=0, linewidth=3):
        """
        Colors the specified edge of the triangle, making the line a bit bolder and leaving some distance from the vertices.
        Parameters:
            edge : Edge to be colored ('AB', 'BC', or 'CA')
        """
        # allow passing a color string instead of a color index
        if isinstance(coloridx, (int, np.integer)):
            offset_factor = coloridx
            color = COLORS[coloridx]
        else:
            offset_factor = 0
            color = coloridx

        edge_offset = self.edge_offset #+ self.edge_offset / 4 #* offset_factor
        match (edge, self.orientation):
            case ("AB", "up"):
                V0 = (self.A[0] + edge_offset, self.A[1])
                V1 = (self.B[0] - edge_offset, self.B[1])
            case ("AB", "down"):
                V0 = (
                    self.A[0] + edge_offset * np.cos(np.pi / 3),
                    self.A[1] + edge_offset * np.sin(np.pi / 3),
                )
                V1 = (
                    self.B[0] - edge_offset * np.cos(np.pi / 3),
                    self.B[1] - edge_offset * np.sin(np.pi / 3),
                )
            case ("BC", "up"):
                V0 = (
                    self.B[0] - edge_offset * np.cos(np.pi / 3),
                    self.B[1] + edge_offset * np.sin(np.pi / 3),
                )
                V1 = (
                    self.C[0] + edge_offset * np.cos(np.pi / 3),
                    self.C[1] - edge_offset * np.sin(np.pi / 3),
                )
            case ("BC", "down"):
                V0 = (self.B[0] - edge_offset, self.B[1])
                V1 = (self.C[0] + edge_offset, self.C[1])
            case ("CA", "up"):
                V0 = (
                    self.C[0] - edge_offset * np.cos(np.pi / 3),
                    self.C[1] - edge_offset * np.sin(np.pi / 3),
                )
                V1 = (
                    self.A[0] + edge_offset * np.cos(np.pi / 3),
                    self.A[1] + edge_offset * np.sin(np.pi / 3),
                )
            case ("CA", "down"):
                V0 = (
                    self.C[0] + edge_offset * np.cos(np.pi / 3),
                    self.C[1] - edge_offset * np.sin(np.pi / 3),
                )
                V1 = (
                    self.A[0] - edge_offset * np.cos(np.pi / 3),
                    self.A[1] + edge_offset * np.sin(np.pi / 3),
                )

        self.ax.plot(
            [V0[0], V1[0]],
            [V0[1], V1[1]],
            color=color,
            linewidth=self.bold_line - offset_factor if linewidth is None else linewidth,
        )

    def color_edges(self, coloridx=0, linewidth=None):
        """
        Colors the edges of the triangle, making the lines a bit bolder and leaving some distance from the vertices.
        """
        self.color_edge("AB", coloridx, linewidth)
        self.color_edge("BC", coloridx, linewidth)
        self.color_edge("CA", coloridx, linewidth)


# ===============================================================================
def draw_arrow(ax, start, end, coloridx=0):
    """
    Draws an arrow from start to end coordinates.
    Parameters:
        ax      : Matplotlib axis object
        start   : Tuple of (x, y) for the start coordinates
        end     : Tuple of (x, y) for the end coordinates
        coloridx: Index for the color in COLORS
    """
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(facecolor=COLORS[coloridx], shrink=0.1, width=1, headwidth=10),
        zorder=99 - coloridx,
    )


# ===============================================================================
def draw_mesh(ax, nx, ny):
    """
    Draws a grid of triangles.
    Parameters:
        ax : Matplotlib axis object
    """
    triangles = []
    for y in range(ny):
        for x in range(nx):
            xA = x * SIDE - y * SIDE / 2
            yA = y * SIDE * np.sqrt(3) / 2
            # Create upward-pointing triangle
            up_triangle = Triangle(ax, xA, yA, "up")
            triangles.append(up_triangle)
            if x < nx - 1:
                # Create downward-pointing triangle
                down_triangle = Triangle(ax, xA + SIDE, yA, "down")
                triangles.append(down_triangle)
        if y < ny - 1:
            # Create the two outer downward-pointing triangles
            xA = -y * SIDE / 2
            down_triangle = Triangle(ax, xA, yA, "down")
            triangles.append(down_triangle)
            xA += nx * SIDE
            down_triangle = Triangle(ax, xA, yA, "down")
            triangles.append(down_triangle)
        nx += 1

    for i, T in enumerate(triangles):
        T.draw()
        if LABEL_TRIANGLES:
            T.print_labels(str(i))

    xlims = (-AX_BORDER - (ny - 1) * SIDE / 2, AX_BORDER + (nx - 1) * SIDE - (ny - 1) * SIDE / 2)
    ylims = (-0.2, AX_BORDER + ny * SIDE * np.sqrt(3) / 2)

    return triangles, xlims, ylims


# ===============================================================================
def add_legend(ax, label, xlims):
    """
    Adds a horizontal legend with rectangles colored with COLORS and labels from the input string.
    Parameters:
        ax    : Matplotlib axis object
        label : String containing the labels for the legend
        xlims : Tuple containing the x-axis limits for centering the legend
    """
    label = label.replace("2", "")  # Remove all '2' from the label
    label = label.replace("o", "")  # Remove possible 'o'
    N = len(label)
    rect_width = 0.2 * SIDE
    rect_height = 0.1 * SIDE
    spacing = 0.4 * SIDE

    # Calculate the starting x position to center the legend
    total_width = N * rect_width + (N - 1) * spacing
    legend_x = (xlims[1] - xlims[0] - total_width) / 2 + xlims[0]
    legend_y = -0.12  # Y position for the legend

    for i in range(N):
        # Draw the rectangle
        rect = plt.Rectangle(
            (legend_x + i * (rect_width + spacing), legend_y),
            rect_width,
            rect_height,
            facecolor=COLORS[i],
        )
        ax.add_patch(rect)

        # Print the label on top of the rectangle
        ax.text(
            legend_x + i * (rect_width + spacing) + rect_width / 2,
            legend_y + rect_height / 2,
            label[i],
            fontsize=12,
            ha="center",
            va="center",
            color="black",
        )

        # Draw the arrow linking to the next rectangle
        if i < N - 1:
            start = (legend_x + i * (rect_width + spacing) + rect_width, legend_y + rect_height / 2)
            end = (legend_x + (i + 1) * (rect_width + spacing), legend_y + rect_height / 2)
            draw_arrow(ax, end, start, i + 1)


# ===============================================================================
def generate_mesh_figure(nx, ny, label, static_dir):
    """
    Generates a figure with a grid of triangles.
    Parameters:
        nx : Number of triangles in the x direction (start raw)
        ny : Number of triangles in the y direction
    """
    fig = plt.figure(1)
    plt.clf()
    plt.show(block=False)
    ax = fig.add_subplot(111)
    T, xlims, ylims = draw_mesh(ax, nx, ny)
    ax.set_title(f"{label}")
    add_legend(ax, label, xlims)
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    ax.set_aspect("equal")
    ax.axis("off")

    figure_dir = os.path.join(static_dir, IMG_DIR)

    fname = os.path.join(figure_dir, f"offsetProvider_{label}.png")
    fig.save = lambda: fig.savefig(fname, dpi=300, bbox_inches="tight")

    return fig, ax, T


# ===============================================================================
def generate_figures(static_dir: str = "."):
    # ---------------------------------------------------------------------------
    # Build c2e as a 2x2 parallelogram, color one down cell blue with orange edges,
    # draw arrows around it, and add placeholder axis arrows/labels for j and i.
    fig, ax, triangles = generate_parallelogram_figure(2, 2, "c2e", static_dir)

    # pick the first down triangle to color (keeps behavior similar to previous T[1])
    target = None
    for key, tri in triangles:
        if key[2] == "down":
            target = tri
            break

    if target is not None:
        # color cell blue (index 0) and edges orange (index 1)
        target.color_cell(0)
        target.color_edges(1)
        # arrows pointing from edges to cell center (same as before)
        draw_arrow(ax, target.AB, target.CC, 1)
        draw_arrow(ax, target.BC, target.CC, 1)
        draw_arrow(ax, target.CA, target.CC, 1)
        # add edge labels (placeholders) at edge midpoints
        # AB
        # ax.text(target.AB[0] + 0.2, target.AB[1], "⟪0,1,0⟫", fontsize=10, ha="center", va="center")
        # # BC
        # ax.text(target.BC[0], target.BC[1] + 0.08, "⟪1,0,-1⟫", fontsize=10, ha="center", va="center")
        # # CA
        # ax.text(target.CA[0] - 0.2, target.CA[1], "⟪0,0,1⟫", fontsize=10, ha="center", va="center")

        # # label the cell center with placeholder
        # ax.text(target.CC[0], target.CC[1], "(0,0,1)", fontsize=10, ha="center", va="center")

    # dimension arrows and placeholder labels
    height = SIDE * np.sqrt(3) / 2
    v1 = (SIDE, 0)
    v2 = (SIDE / 2, height)
    x0, y0 = 0.0, 0.0

    P0 = (x0, y0)
    P1 = (x0 + 2 * v1[0], y0 + 2 * v1[1])
    P3 = (x0 + 2 * v2[0], y0 + 2 * v2[1])

    # label vertices with dimension placeholders (no arrows)
    # bottom-left origin
    ax.text(P0[0] - 0.03, P0[1] - 0.03, "(i, j) = (0,0)", fontsize=10, ha="right", va="top")
    # far-right corner along j
    ax.text(P1[0] + 0.03, P1[1] - 0.03, "(i, j) = (0,2)", fontsize=10, ha="left", va="top")
    # top corner along i
    ax.text(P3[0] - 0.03, P3[1], "(i, j) = (2,0)", fontsize=10, ha="right", va="bottom")

    fig.savefig(os.path.join(static_dir, IMG_DIR, "offsetProvider_c2e.png"), dpi=300, bbox_inches="tight")

    # ---------------------------------------------------------------------------
    # fig, ax, T = generate_mesh_figure(2, 2, "c2e2c", static_dir)

    # Ta = T[1]
    # Tb = T[0]
    # Tc = T[2]
    # Td = T[7]
    # Ta.color_cell()
    # draw_arrow(ax, Ta.AB, Ta.CC, 1)
    # draw_arrow(ax, Ta.BC, Ta.CC, 1)
    # draw_arrow(ax, Ta.CA, Ta.CC, 1)
    # Ta.color_edges(1)
    # draw_arrow(ax, Tb.CC, Tb.BC, 2)
    # draw_arrow(ax, Tc.CC, Tc.CA, 2)
    # draw_arrow(ax, Td.CC, Td.AB, 2)
    # Tb.color_cell(2)
    # Tc.color_cell(2)
    # Td.color_cell(2)

    # fig.save()

    # # ---------------------------------------------------------------------------
    # fig, ax, T = generate_mesh_figure(2, 2, "c2e2co", static_dir)

    # Ta = T[1]
    # Tb = T[0]
    # Tc = T[2]
    # Td = T[7]
    # Ta.color_cell()
    # draw_arrow(ax, Ta.AB, Ta.CC, 1)
    # draw_arrow(ax, Ta.BC, Ta.CC, 1)
    # draw_arrow(ax, Ta.CA, Ta.CC, 1)
    # Ta.color_edges(1)
    # draw_arrow(ax, Tb.CC, Tb.BC, 2)
    # draw_arrow(ax, Tc.CC, Tc.CA, 2)
    # draw_arrow(ax, Td.CC, Td.AB, 2)
    # Tb.color_cell(2)
    # Tc.color_cell(2)
    # Td.color_cell(2)
    # draw_arrow(ax, Ta.CC, Ta.AB, 2)
    # draw_arrow(ax, Ta.CC, Ta.BC, 2)
    # draw_arrow(ax, Ta.CC, Ta.CA, 2)
    # Ta.color_cell(2)

    # fig.save()

    # # ---------------------------------------------------------------------------
    # # ---------------------------------------------------------------------------
    # fig, ax, T = generate_mesh_figure(2, 2, "e2v", static_dir)

    # Ta = T[1]
    # Ta.color_edge("BC")
    # draw_arrow(ax, Ta.B, Ta.BC, 1)
    # draw_arrow(ax, Ta.C, Ta.BC, 1)
    # Ta.color_vertex("B", 1)
    # Ta.color_vertex("C", 1)

    # fig.save()

    # # ---------------------------------------------------------------------------
    # fig, ax, T = generate_mesh_figure(2, 2, "e2c", static_dir)

    # Ta = T[1]
    # Tb = T[7]
    # Ta.color_edge("BC")
    # draw_arrow(ax, Ta.CC, Ta.BC, 1)
    # draw_arrow(ax, Tb.CC, Tb.AB, 1)
    # Ta.color_cell(1)
    # Tb.color_cell(1)

    # fig.save()

    # # ---------------------------------------------------------------------------
    # fig, ax, T = generate_mesh_figure(2, 2, "e2c2e", static_dir)

    # Ta = T[1]
    # Tb = T[7]
    # Ta.color_edge("BC")
    # draw_arrow(ax, Ta.CC, Ta.BC, 1)
    # draw_arrow(ax, Tb.CC, Tb.AB, 1)
    # Ta.color_cell(1)
    # Tb.color_cell(1)
    # draw_arrow(ax, Ta.AB, Ta.CC, 2)
    # draw_arrow(ax, Ta.BC, Ta.CC, 2)
    # draw_arrow(ax, Ta.CA, Ta.CC, 2)
    # draw_arrow(ax, Tb.AB, Tb.CC, 2)
    # draw_arrow(ax, Tb.BC, Tb.CC, 2)
    # draw_arrow(ax, Tb.CA, Tb.CC, 2)
    # Ta.color_edges(2)
    # Tb.color_edges(2)

    # fig.save()

    # # ---------------------------------------------------------------------------
    # fig, ax, T = generate_mesh_figure(2, 2, "e2c2v", static_dir)

    # Ta = T[1]
    # Tb = T[7]
    # Ta.color_edge("BC")
    # draw_arrow(ax, Ta.CC, Ta.BC, 1)
    # draw_arrow(ax, Tb.CC, Tb.AB, 1)
    # Ta.color_cell(1)
    # Tb.color_cell(1)
    # draw_arrow(ax, Ta.A, Ta.CC, 2)
    # draw_arrow(ax, Ta.B, Ta.CC, 2)
    # draw_arrow(ax, Ta.C, Ta.CC, 2)
    # draw_arrow(ax, Tb.A, Tb.CC, 2)
    # draw_arrow(ax, Tb.B, Tb.CC, 2)
    # draw_arrow(ax, Tb.C, Tb.CC, 2)
    # Ta.color_vertices(2)
    # Tb.color_vertices(2)

    # fig.save()

    # # ---------------------------------------------------------------------------
    # # ---------------------------------------------------------------------------
    # fig, ax, T = generate_mesh_figure(1, 2, "v2e", static_dir)

    # Ta = T[0]
    # Tb = T[2]
    # Tc = T[5]
    # Td = T[4]
    # Te = T[3]
    # Tf = T[1]
    # Ta.color_vertex("C")
    # draw_arrow(ax, Ta.CA, Ta.C, 1)
    # draw_arrow(ax, Tb.CA, Tb.C, 1)
    # draw_arrow(ax, Tc.AB, Tc.A, 1)
    # draw_arrow(ax, Td.AB, Td.A, 1)
    # draw_arrow(ax, Te.BC, Te.B, 1)
    # draw_arrow(ax, Tf.BC, Tf.B, 1)
    # Ta.color_edge("CA", 1)
    # Tb.color_edge("CA", 1)
    # Tc.color_edge("AB", 1)
    # Td.color_edge("AB", 1)
    # Te.color_edge("BC", 1)
    # Tf.color_edge("BC", 1)

    # fig.save()


# ==============================================================================
def draw_unit_parallelogram(ax, x0, y0, color=None):
    """
    Draws a single unit parallelogram starting at (x0, y0).
    The parallelogram is defined by vectors v1=(SIDE,0) and v2=(SIDE/2, height).
    Returns the list of corner points.
    """
    height = SIDE * np.sqrt(3) / 2
    v1 = (SIDE, 0)
    v2 = (SIDE / 2, height)

    p0 = (x0, y0)
    p1 = (x0 + v1[0], y0 + v1[1])
    p2 = (p1[0] + v2[0], p1[1] + v2[1])
    p3 = (x0 + v2[0], y0 + v2[1])

    poly = plt.Polygon([p0, p1, p2, p3], closed=True, edgecolor="black", facecolor="none")
    ax.add_patch(poly)
    return [p0, p1, p2, p3]


# ==============================================================================
def draw_parallelogram_grid(ax, nx, ny, x0=0.0, y0=0.0):
    """
    Draws a grid of unit parallelograms of size nx (horizontal) by ny (vertical).
    Also draws a bold outer boundary to make the full edges visible.
    """
    height = SIDE * np.sqrt(3) / 2
    v1 = (SIDE, 0)
    v2 = (SIDE / 2, height)

    # draw all unit parallelograms as pairs of triangles
    triangles = []
    for j in range(ny):
        for i in range(nx):
            ox = x0 + i * v1[0] + j * v2[0]
            oy = y0 + i * v1[1] + j * v2[1]
            # up triangle at (ox,oy)
            up = Triangle(ax, ox, oy, "up")
            up.draw()
            triangles.append(((i, j, "up"), up))
            # down triangle to the right of up triangle
            down = Triangle(ax, ox + SIDE, oy, "down")
            down.draw()
            triangles.append(((i, j, "down"), down))

    # draw outer boundary (large parallelogram)
    P0 = (x0, y0)
    P1 = (x0 + nx * v1[0], y0 + nx * v1[1])
    P2 = (P1[0] + ny * v2[0], P1[1] + ny * v2[1])
    P3 = (x0 + ny * v2[0], y0 + ny * v2[1])

    ax.plot([P0[0], P1[0], P2[0], P3[0], P0[0]], [P0[1], P1[1], P2[1], P3[1], P0[1]],
            color="black", linewidth=1)

    # compute reasonable limits
    minx = min(P0[0], P1[0], P2[0], P3[0]) - AX_BORDER
    maxx = max(P0[0], P1[0], P2[0], P3[0]) + AX_BORDER
    miny = min(P0[1], P1[1], P2[1], P3[1]) - AX_BORDER
    maxy = max(P0[1], P1[1], P2[1], P3[1]) + AX_BORDER

    return triangles, (minx, maxx), (miny, maxy)


# ==============================================================================
def generate_parallelogram_figure(nx: int, ny: int, label: str = None, static_dir: str = "."):
    """
    Generates and saves a figure showing a parallelogram grid.
    """
    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    triangles, xlims, ylims = draw_parallelogram_grid(ax, nx, ny, x0=0.0, y0=0.0)
    ax.set_title(f"Parallelogram grid: {label}" if label else None)
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    ax.set_aspect("equal")
    ax.axis("off")

    figure_dir = os.path.join(static_dir, IMG_DIR)
    os.makedirs(figure_dir, exist_ok=True)
    fname = os.path.join(figure_dir, f"offsetProvider_parallelogram_{label}.png")
    fig.savefig(fname, dpi=300, bbox_inches="tight")
    return fig, ax, triangles


# ==============================================================================
def generate_unit_cell_figure(static_dir: str = "."):
    """
    Generate and save a PNG showing a single unit cell: the up-triangle
    (vertex lower-left) with its three edges, and the adjacent down-triangle
    filled but without its edges/vertices.
    """
    height = SIDE * np.sqrt(3) / 2
    x0, y0 = 0.0, 0.0
    v1 = (SIDE, 0)
    v2 = (SIDE / 2, height)

    p0 = (x0, y0)
    p1 = (x0 + v1[0], y0 + v1[1])
    p2 = (p1[0] + v2[0], p1[1] + v2[1])
    p3 = (x0 + v2[0], y0 + v2[1])

    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)

    # Colors: cells blue, vertices green, edges orange
    cell_color = "tab:blue"
    vertex_color = "tab:green"
    edge_color = "tab:orange"

    # Use Triangle helpers so coloring functions are consistent
    up = Triangle(ax, p0[0], p0[1], "up")
    down = Triangle(ax, p1[0], p1[1], "down")

    # draw base outlines
    up.draw()
    down.draw()

    # fill cells with color index 0 (blue), color edges with 1 (orange), vertices with 2 (green)
    up.color_cell(0)
    up.color_edges(1, linewidth=5)
    up.color_vertex("A", 2)

    # fill the down triangle only (no edges/vertices)
    down.color_cell(0)

    # set limits
    ax.set_xlim(-AX_BORDER, p2[0] + AX_BORDER)
    ax.set_ylim(-0.2, p2[1] + AX_BORDER)
    ax.set_aspect("equal")
    ax.axis("off")

    figure_dir = os.path.join(static_dir, IMG_DIR)
    os.makedirs(figure_dir, exist_ok=True)
    fname = os.path.join(figure_dir, "offsetProvider_unit_cell.png")
    fig.savefig(fname, dpi=300, bbox_inches="tight")
    return fig, ax


# ==============================================================================
def generate_parallelogram_with_colored_boundary(nx: int, ny: int, label: str = None, static_dir: str = ".", color="red"):
    """
    Generate a parallelogram grid image and recolor the outer boundary edges
    (the edges added after tiling unit cells) with `color`.
    """
    fig, ax, triangles = generate_parallelogram_figure(nx, ny, label, static_dir)

    # build vertex dictionary: (i,j) -> (x,y), with i=0..nx, j=0..ny
    height = SIDE * np.sqrt(3) / 2
    v1 = (SIDE, 0)
    v2 = (SIDE / 2, height)
    x0, y0 = 0.0, 0.0
    verts = {}
    for j in range(ny + 1):
        for i in range(nx + 1):
            vx = x0 + i * v1[0] + j * v2[0]
            vy = y0 + i * v1[1] + j * v2[1]
            verts[(i, j)] = (vx, vy)

    # helper to find a triangle and its edge name for a vertex pair
    def find_triangle_edge(pA, pB):
        for (_, tri) in triangles:
            for edge in ("AB", "BC", "CA"):
                coords = {"AB": (tri.A, tri.B), "BC": (tri.B, tri.C), "CA": (tri.C, tri.A)}[edge]
                # use allclose to avoid floating-point mismatches
                if (np.allclose(coords[0], pA) and np.allclose(coords[1], pB)) or (
                    np.allclose(coords[0], pB) and np.allclose(coords[1], pA)
                ):
                    return tri, edge
        return None, None

    # color topmost edges (j = ny) in blue with linewidth 3
    blue_idx = 3
    for i in range(nx):
        A = verts[(i, ny)]
        B = verts[(i + 1, ny)]
        tri, edge = find_triangle_edge(A, B)
        if tri is not None:
            tri.color_edge(edge, blue_idx, linewidth=2)

    # color rightmost edges (i = nx) in blue with linewidth 3
    for j in range(ny):
        A = verts[(nx, j)]
        B = verts[(nx, j + 1)]
        tri, edge = find_triangle_edge(A, B)
        if tri is not None:
            tri.color_edge(edge, blue_idx, linewidth=2)

    # color vertices along top row and right column (use color index 3)
    def find_triangle_with_vertex(p):
        for (_, tri) in triangles:
            if tri.A == p or tri.B == p or tri.C == p:
                return tri
        return None

    # color the vertices along top row and right column using blue index
    for i in range(nx + 1):
        p = verts[(i, ny)]
        tri = find_triangle_with_vertex(p)
        if tri is not None:
            if tri.A == p:
                tri.color_vertex("A", blue_idx)
            elif tri.B == p:
                tri.color_vertex("B", blue_idx)
            else:
                tri.color_vertex("C", blue_idx)
    for j in range(ny + 1):
        p = verts[(nx, j)]
        tri = find_triangle_with_vertex(p)
        if tri is not None:
            if tri.A == p:
                tri.color_vertex("A", blue_idx)
            elif tri.B == p:
                tri.color_vertex("B", blue_idx)
            else:
                tri.color_vertex("C", blue_idx)

    figure_dir = os.path.join(static_dir, IMG_DIR)
    fname = os.path.join(figure_dir, f"offsetProvider_parallelogram_{label}_boundary_{color}.png")
    fig.savefig(fname, dpi=300, bbox_inches="tight")
    return fig, ax


# ==============================================================================
def generate_parallelogram_colored_loops(nx: int, ny: int, loops: int, label: str, static_dir: str = "."):
    """
    Generate a parallelogram grid of size nx x ny and overlay up to `loops`
    concentric edge-loops, coloring each loop with a different color.
    """
    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    triangles, _, _ = draw_parallelogram_grid(ax, nx, ny, x0=0.0, y0=0.0)

    height = SIDE * np.sqrt(3) / 2
    v1 = (SIDE, 0)
    v2 = (SIDE / 2, height)
    x0, y0 = 0.0, 0.0

    max_loops = min(nx, ny) // 2
    loops = min(loops, max_loops)

    palette = COLORS
    intermediate_color = "#8A2BE2"  # purple for intermediate layer (fallback)
    # helper: return a lighter version of a color (hex or named)
    def lighten_color(col, amount=0.5):
        try:
            rgb = mcolors.to_rgb(col)
        except Exception:
            rgb = (0.5, 0.5, 0.5)
        lighter = tuple(c + (1.0 - c) * amount for c in rgb)
        return mcolors.to_hex(lighter)

    # build vertex dictionary: (i,j) -> (x,y), with i=0..nx, j=0..ny
    verts = {}
    for j in range(ny + 1):
        for i in range(nx + 1):
            vx = x0 + i * v1[0] + j * v2[0]
            vy = y0 + i * v1[1] + j * v2[1]
            verts[(i, j)] = (vx, vy)

    # palette for connector layers (edges joining two consecutive loops)
    CONNECTOR_COLORS = ["#8A2BE2", "#7FFFD4", "#FFD700", "#FF69B4", "#00CED1", "#ADFF2F"]

    # map triangle keys to objects for quick lookup: ((i,j,orient) -> Triangle)
    tri_map = {key: tri for (key, tri) in triangles}

    # helper to find triangle edge for a vertex pair
    def find_triangle_edge(pA, pB):
        for (_, tri) in triangles:
            for edge in ("AB", "BC", "CA"):
                coords = {"AB": (tri.A, tri.B), "BC": (tri.B, tri.C), "CA": (tri.C, tri.A)}[edge]
                if (np.allclose(coords[0], pA) and np.allclose(coords[1], pB)) or (
                    np.allclose(coords[0], pB) and np.allclose(coords[1], pA)
                ):
                    return tri, edge
        return None, None

    for k in range(loops):
        i0 = k
        j0 = k
        i1 = nx - k
        j1 = ny - k

        # enumerate edges along the four sides and color via Triangle.color_edge
        # top side: (i=i0..i1-1, j=j1)
        for i in range(i0, i1):
            A = (x0 + i * v1[0] + j1 * v2[0], y0 + i * v1[1] + j1 * v2[1])
            B = (x0 + (i + 1) * v1[0] + j1 * v2[0], y0 + (i + 1) * v1[1] + j1 * v2[1])
            tri, edge = find_triangle_edge(A, B)
            if tri is not None:
                tri.color_edge(edge, k % len(palette), linewidth=1.5)
            else:
                # fallback: draw the colored line directly to ensure coverage
                ax.plot([A[0], B[0]], [A[1], B[1]], color=COLORS[k % len(palette)], linewidth=3)

        # right side: (j=j0..j1-1, i=i1)
        for j in range(j0, j1):
            A = (x0 + i1 * v1[0] + j * v2[0], y0 + i1 * v1[1] + j * v2[1])
            B = (x0 + i1 * v1[0] + (j + 1) * v2[0], y0 + i1 * v1[1] + (j + 1) * v2[1])
            tri, edge = find_triangle_edge(A, B)
            if tri is not None:
                tri.color_edge(edge, k % len(palette), linewidth=1.5)
            else:
                ax.plot([A[0], B[0]], [A[1], B[1]], color=COLORS[k % len(palette)], linewidth=3)

        # bottom side: (i=i0..i1-1, j=j0)
        for i in range(i0, i1):
            A = (x0 + i * v1[0] + j0 * v2[0], y0 + i * v1[1] + j0 * v2[1])
            B = (x0 + (i + 1) * v1[0] + j0 * v2[0], y0 + (i + 1) * v1[1] + j0 * v2[1])
            tri, edge = find_triangle_edge(A, B)
            if tri is not None:
                tri.color_edge(edge, k % len(palette), linewidth=1.5)
            else:
                ax.plot([A[0], B[0]], [A[1], B[1]], color=COLORS[k % len(palette)], linewidth=3)

        # left side: (j=j0..j1-1, i=i0)
        for j in range(j0, j1):
            A = (x0 + i0 * v1[0] + j * v2[0], y0 + i0 * v1[1] + j * v2[1])
            B = (x0 + i0 * v1[0] + (j + 1) * v2[0], y0 + i0 * v1[1] + (j + 1) * v2[1])
            tri, edge = find_triangle_edge(A, B)
            if tri is not None:
                tri.color_edge(edge, k % len(palette), linewidth=1.5)
            else:
                ax.plot([A[0], B[0]], [A[1], B[1]], color=COLORS[k % len(palette)], linewidth=3)

        # color connector edges between layer k and k+1 (all edge orientations)
        if k < loops:
            # derive connector color as a lighter version of the outer loop color
            outer_color = palette[k % len(palette)]
            conn_color = lighten_color(outer_color, amount=0.6)

            # horizontal edges: (i,i+1) at constant j
            for j in range(0, ny + 1):
                for i in range(0, nx):
                    u = (i, j)
                    v = (i + 1, j)
                    lu = min(u[0], u[1], nx - u[0], ny - u[1])
                    lv = min(v[0], v[1], nx - v[0], ny - v[1])
                    if {lu, lv} == {k, k + 1}:
                        A = verts[u]
                        B = verts[v]
                        tri, edge = find_triangle_edge(A, B)
                        if tri is not None:
                            tri.color_edge(edge, conn_color, linewidth=2)
                        else:
                            ax.plot([A[0], B[0]], [A[1], B[1]], color=conn_color, linewidth=2)

            # vertical edges: (j,j+1) at constant i
            for i in range(0, nx + 1):
                for j in range(0, ny):
                    u = (i, j)
                    v = (i, j + 1)
                    lu = min(u[0], u[1], nx - u[0], ny - u[1])
                    lv = min(v[0], v[1], nx - v[0], ny - v[1])
                    if {lu, lv} == {k, k + 1}:
                        A = verts[u]
                        B = verts[v]
                        tri, edge = find_triangle_edge(A, B)
                        if tri is not None:
                            tri.color_edge(edge, conn_color, linewidth=2)
                        else:
                            ax.plot([A[0], B[0]], [A[1], B[1]], color=conn_color, linewidth=2)

            # diagonal edges: (i+1,j) -- (i,j+1)
            for i in range(0, nx):
                for j in range(0, ny):
                    u = (i + 1, j)
                    v = (i, j + 1)
                    lu = min(u[0], u[1], nx - u[0], ny - u[1])
                    lv = min(v[0], v[1], nx - v[0], ny - v[1])
                    # normal connector between consecutive layers
                    cond1 = {lu, lv} == {k, k + 1}
                    # special-case corner diagonals: both endpoints report layer k
                    # but the diagonal lies between loop k and k+1 (bottom-left or top-right)
                    cond2 = (lu == lv == k) and (
                        (i == i0 and j == j0) or (i == i1 - 1 and j == j1 - 1)
                    )
                    if cond1 or cond2:
                        A = verts[u]
                        B = verts[v]
                        tri, edge = find_triangle_edge(A, B)
                        if tri is not None:
                            tri.color_edge(edge, conn_color, linewidth=2)
                        else:
                            ax.plot([A[0], B[0]], [A[1], B[1]], color=conn_color, linewidth=2)

        # collect this loop's vertex points so we can color vertices
        loop_vertices = set()
        for i in range(i0, i1 + 1):
            loop_vertices.add(verts[(i, j1)])
            loop_vertices.add(verts[(i, j0)])
        for j in range(j0, j1 + 1):
            loop_vertices.add(verts[(i0, j)])
            loop_vertices.add(verts[(i1, j)])

        # color vertices for this loop
        for p in loop_vertices:
            for (key, tri) in triangles:
                if np.allclose(tri.A, p):
                    tri.color_vertex("A", k % len(palette))
                    break
                if np.allclose(tri.B, p):
                    tri.color_vertex("B", k % len(palette))
                    break
                if np.allclose(tri.C, p):
                    tri.color_vertex("C", k % len(palette))
                    break

        # color entire unit parallelograms (both up/down triangles) for this layer
        for i in range(i0, i1):
            for j in range(j0, j1):
                # compute unit cell layer
                unit_layer = min(i, j, nx - 1 - i, ny - 1 - j)
                if unit_layer == k:
                    up_key = (i, j, "up")
                    down_key = (i, j, "down")
                    if up_key in tri_map:
                        tri_map[up_key].color_cell(k % len(palette))
                    if down_key in tri_map:
                        tri_map[down_key].color_cell(k % len(palette))

    ax.set_aspect("equal")
    ax.axis("off")
    figure_dir = os.path.join(static_dir, IMG_DIR)
    os.makedirs(figure_dir, exist_ok=True)
    fname = os.path.join(figure_dir, f"offsetProvider_parallelogram_{label}_loops_{loops}.png")
    fig.savefig(fname, dpi=300, bbox_inches="tight")
    return fig, ax


# ===============================================================================
def generate_page(static_dir: str):
    """
    Generates a documentation page collecting all the figures created by
    `generate_figures`. The figures are sorted in alphabetical order.
    """
    figure_dir = os.path.join(static_dir, IMG_DIR)

    figure_paths = [
        os.path.join(figure_dir, f)
        for f in os.listdir(figure_dir)
        if "offsetProvider_" in f and f.endswith(".png")
    ]
    figure_paths.sort()

    # TODO(): use constant for _source location
    page_rst_path = os.path.join("_source", "offset_providers.rst")

    with open(page_rst_path, "w") as f:
        f.write("Offset providers\n")
        f.write("================\n\n")
        f.write("This page contains the figures for the offset providers.\n\n")
        for fig_path in figure_paths:
            relative_path = os.path.relpath(fig_path, os.path.dirname(page_rst_path))

            label = fig_path.replace("offsetProvider_", "").replace(".png", "")
            f.write(f".. image:: {relative_path}\n")
            f.write("   :align: center\n")
            f.write(f"   :alt: {label}\n")
            f.write("   :class: offset-provider-img\n")


# ===============================================================================
if __name__ == "__main__":
    # generate the existing offset provider figures
    generate_figures()

    # generate unit cell image
    # generate_unit_cell_figure(static_dir=".")

    # generate a 10x8 parallelogram and a recolored-boundary copy
    # generate_parallelogram_figure(10, 8, static_dir=".")
    # generate_parallelogram_with_colored_boundary(10, 8, static_dir=".", color="red")

    # generate a larger 20x15 with 5 colored loops
    # generate_parallelogram_colored_loops(20, 15, 5, "20x15", static_dir=".")

    # plt.show()
