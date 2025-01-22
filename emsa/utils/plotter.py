import os

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt, colors
from matplotlib.cm import ScalarMappable
from matplotlib.tri import Triangulation


def visualize_transmission_graph(state_data, trans_data, tms_rules):
    plt.figure(figsize=(10, 8))
    G = nx.DiGraph()

    # Add nodes
    def get_node_label(node):
        if (n_substates := state_data[node].get("n_substates", 1)) > 1:
            return f"{node}$^{{{n_substates}}}$"
        return node

    def get_node_color(node):  # TODO: color dict based
        node_type = state_data[node].get("type", "basic")
        if node_type == "susceptible":
            return "green"
        if node_type == "infected":
            return "red"
        if node_type == "recovered":
            return "blue"
        if node_type == "dead":
            return "gray"
        return "orange"

    for node, data in state_data.items():
        G.add_node(get_node_label(node), color=get_node_color(node))

    # Add edges from transitions
    def get_edge_label(trans):
        if trans.get("params"):
            distr_muls = [
                distr if distr[-1] != "_" else f"(1 - {distr[:-1]})" for distr in trans["params"]
            ]
            return " * ".join(distr_muls + [state_data[trans["source"]]["rate"]])
        return state_data[trans["source"]].get("rate", "")

    trans_edges = [
        (
            get_node_label(transition["source"]),
            get_node_label(transition["target"]),
            get_edge_label(transition),
        )
        for transition in trans_data
    ]
    for edge in trans_edges:
        source, target, param = edge
        G.add_edge(source, target, param=param)

    # Add transmission rules as edges
    tms_edges = []
    tms_edges_unpar = []
    for rule in tms_rules:
        source = get_node_label(rule["source"])
        target = get_node_label(rule["target"])
        G.add_edge(source, target, param="beta")  # TODO: susc mul, inf mul
        for actor, param in rule["actors-params"].items():
            actor = get_node_label(actor)
            if param is not None:
                tms_edges.append((actor, source, param))
            else:
                tms_edges_unpar.append((actor, source))

    if len(state_data) > 4:
        pos = nx.kamada_kawai_layout(G)
        pos = nx.arf_layout(G, pos=pos)
    else:
        pos = nx.circular_layout(G)

    node_colors = [node[1]["color"] for node in G.nodes(data=True)]
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_size=500,
        node_color=node_colors,
        font_size=12,
        arrowsize=10,
    )

    # Draw full edges with labels
    labels = nx.get_edge_attributes(G, "param")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_size=7)

    # Draw dashed edges and label
    dashed_edge_labels = {(edge[0], edge[1]): edge[2] for edge in tms_edges}
    nx.draw_networkx_edges(G, pos, edgelist=tms_edges + tms_edges_unpar, width=1, style=":")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=dashed_edge_labels, font_size=7)

    plt.title("Transmission Graph")
    plt.show()


def generate_tornado_plot(
    sim_object, labels, prcc: np.ndarray, p_val, filename: str, title=None, verbose=True
):
    """
    Generate a tornado plot to visualize the Partial Rank Correlation Coefficient (PRCC) values.
    """
    prcc = np.round(prcc, 3)
    prcc = np.atleast_1d(prcc)
    ys = range(len(labels))[::-1]

    p_val_colors = ["green", "yellow", "red"]
    thresholds = [0, 0.01, 0.1, 1]
    cmap = colors.ListedColormap(p_val_colors)
    norm = colors.BoundaryNorm(boundaries=thresholds, ncolors=cmap.N, clip=True)

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, orientation="vertical", shrink=0.7, pad=0.02, ax=plt.gca())
    cbar.set_label("p-values")

    # Plot the bars one by one
    for y, value in zip(ys, prcc):
        p = p_val[len(prcc) - 1 - y] if p_val.ndim > 0 else p_val
        facecolor = cmap(norm((p)))
        plt.broken_barh(
            [(value if value < 0 else 0, abs(value))],
            (y - 0.4, 0.8),
            facecolors=[facecolor, facecolor],
            edgecolors=["black", "black"],
            linewidth=1,
        )

        if value != 0:
            x = (value / 2) if np.abs(value) >= 0.2 else (-np.sign(value) * 0.1)
        else:
            x = -0.1
        plt.text(x, y - 0.01, str(value), va="center", ha="center")

    plt.axvline(0, color="black")

    # Position the x-axis on the top, hide all the other spines (=axis lines)
    axes = plt.gca()  # (gca = get current axes)
    axes.spines["left"].set_visible(False)
    axes.spines["right"].set_visible(False)
    axes.spines["bottom"].set_visible(False)
    axes.xaxis.set_ticks_position("top")

    # Display age groups next to y axis

    plt.yticks(ys, labels)

    # Set the portion of the x- and y-axes to show
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1, len(labels))

    if title is not None:
        plt.title(title)
    if verbose and title is None:
        plt.title(f"PRCC values for {filename}")
    plot_path = os.path.join(sim_object.folder_name, f"prcc_plots/prcc_tornado_plot_{filename}.pdf")
    plt.savefig(plot_path, format="pdf", bbox_inches="tight")
    plt.show()


def get_age_groups():
    return [
        f"{5 * age_start}-{5 * age_start + 5}" if 5 * age_start != 75 else "75+"
        for age_start in range(16)
    ]


def construct_triangle_grid(n_age):
    # vertices of the little squares
    xv, yv = np.meshgrid(np.arange(-0.5, n_age), np.arange(-0.5, n_age))
    # centers of the little square
    xc, yc = np.meshgrid(np.arange(0, n_age), np.arange(0, n_age))
    x = np.concatenate([xv.ravel(), xc.ravel()])
    y = np.concatenate([yv.ravel(), yc.ravel()])
    triangles_prcc = [
        (i + j * (n_age + 1), i + 1 + j * (n_age + 1), i + (j + 1) * (n_age + 1))
        for j in range(n_age)
        for i in range(n_age)
    ]
    triangles_p = [
        (
            i + 1 + j * (n_age + 1),
            i + 1 + (j + 1) * (n_age + 1),
            i + (j + 1) * (n_age + 1),
        )
        for j in range(n_age)
        for i in range(n_age)
    ]
    triang = [
        Triangulation(x, y, triangles, mask=None) for triangles in [triangles_prcc, triangles_p]
    ]
    return triang


def get_prcc_and_p_values(n_age, prcc_vector, p_values):
    from emsa_examples.contact_sensitivity.sensitivity_model_contact import (
        get_rectangular_matrix_from_upper_triu,
    )

    prcc_mtx = np.array(
        get_rectangular_matrix_from_upper_triu(rvector=prcc_vector, matrix_size=n_age)
    )
    p_values_mtx = np.array(
        get_rectangular_matrix_from_upper_triu(rvector=p_values, matrix_size=n_age)
    )
    values = np.array([prcc_mtx, p_values_mtx])
    return values


def plot_prcc_p_values_as_heatmap(n_age, prcc_vector, p_values, filename_to_save, plot_title):
    p_value_cmap = colors.ListedColormap(["Orange", "red", "darkred"])
    cmaps = ["Greens", p_value_cmap]

    thresholds = [0, 0.01, 0.1, 1]
    log_norm = colors.BoundaryNorm(
        boundaries=thresholds, ncolors=p_value_cmap.N, clip=True
    )  # used for p_values
    norm = plt.Normalize(vmin=0, vmax=1)  # used for PRCC_values

    fig, ax = plt.subplots()
    triang = construct_triangle_grid(n_age=n_age)
    mask = get_prcc_and_p_values(n_age=n_age, prcc_vector=prcc_vector, p_values=p_values)
    images = [
        ax.tripcolor(t, np.ravel(val), cmap=cmap, ec="white")
        for t, val, cmap in zip(triang, mask, cmaps)
    ]

    fig.colorbar(images[0], ax=ax, shrink=0.7, aspect=20 * 0.7, label="PRCC")  # for the prcc values
    fig.colorbar(
        images[1], ax=ax, shrink=0.7, aspect=20 * 0.7, pad=0.1, label="p-value"
    )  # p-values

    images[1].set_norm(norm=log_norm)
    images[0].set_norm(norm=norm)

    ax.set_xticks(range(n_age))
    ax.set_yticks(range(n_age))
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax.set(frame_on=False)
    plt.gca().grid(which="minor", color="gray", linestyle="-", linewidth=1)
    ax.margins(x=0, y=0)
    ax.set_aspect("equal", "box")
    plt.tight_layout()
    plt.title(plot_title, y=1.03, fontsize=25)
    plt.savefig(filename_to_save, format="pdf", bbox_inches="tight")
    plt.show()
    plt.close()
