import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import math
import time  # For pausing between scenes conceptually
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, Arrow, FancyArrowPatch
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei"]  # Use SimHei font
plt.rcParams["axes.unicode_minus"] = False  # Fix minus sign display issue

# Configure Chinese font
font_path = "/data_ext/gradio_model_mix/data/SimHei.ttf"
fm.fontManager.addfont(font_path)
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = [fm.FontProperties(fname=font_path).get_name()]
plt.rcParams["axes.unicode_minus"] = False
# --- Animation Setup ---
fig, ax = plt.subplots(figsize=(10, 6))
plt.style.use("seaborn-v0_8-darkgrid")  # Use a nice style

# --- Constants and Parameters ---
TOTAL_FRAMES = 600  # Total frames for the entire animation
FPS = 15  # Frames per second

# Scene timings (approximate frame counts)
SCENE_INTRO_FRAMES = 60
SCENE_BASIC_ATTN_FRAMES = 150
SCENE_MHA_FRAMES = 130
SCENE_MQA_FRAMES = 100
SCENE_GQA_FRAMES = 100
SCENE_OUTRO_FRAMES = 60

# Ensure total frames match sum of scene frames (or adjust scenes)
assert TOTAL_FRAMES == (
    SCENE_INTRO_FRAMES
    + SCENE_BASIC_ATTN_FRAMES
    + SCENE_MHA_FRAMES
    + SCENE_MQA_FRAMES
    + SCENE_GQA_FRAMES
    + SCENE_OUTRO_FRAMES
)

# Colors
COLOR_Q = "#1f77b4"  # Blue
COLOR_K = "#ff7f0e"  # Orange
COLOR_V = "#2ca02c"  # Green
COLOR_SCORES = "#d62728"  # Red
COLOR_OUTPUT = "#9467bd"  # Purple
COLOR_TEXT = "#333333"
COLOR_HEAD_1 = "#aec7e8"
COLOR_HEAD_2 = "#ffbb78"
COLOR_HEAD_3 = "#98df8a"
COLOR_HEAD_4 = "#ff9896"
COLOR_SHARED = "#c7c7c7"  # Grey for shared K/V

# --- Helper Functions for Drawing ---


def draw_box(
    ax,
    center_x,
    center_y,
    width,
    height,
    label,
    color,
    text_color=COLOR_TEXT,
    alpha=1.0,
    zorder=1,
):
    """Draws a labeled box."""
    rect = patches.Rectangle(
        (center_x - width / 2, center_y - height / 2),
        width,
        height,
        linewidth=1,
        edgecolor="black",
        facecolor=color,
        alpha=alpha,
        zorder=zorder,
    )
    ax.add_patch(rect)
    ax.text(
        center_x,
        center_y,
        label,
        ha="center",
        va="center",
        color=text_color,
        fontsize=10,
        zorder=zorder + 1,
    )
    return rect


def draw_arrow(
    ax, start_pos, end_pos, color="black", alpha=1.0, mutation_scale=15, zorder=0
):
    """Draws an arrow between two points."""
    arrow = patches.FancyArrowPatch(
        start_pos,
        end_pos,
        arrowstyle="->",
        mutation_scale=mutation_scale,
        color=color,
        alpha=alpha,
        zorder=zorder,
        linewidth=1.5,
    )
    ax.add_patch(arrow)
    return arrow


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
    return e_x / e_x.sum()


# --- Animation Update Function ---


def update(frame):
    ax.clear()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    t = frame / TOTAL_FRAMES  # Normalized time [0, 1]

    # --- Scene 1: Introduction (Frames 0 to SCENE_INTRO_FRAMES-1) ---
    if frame < SCENE_INTRO_FRAMES:
        progress = frame / SCENE_INTRO_FRAMES
        ax.text(
            5,
            4.5,
            "Core of Large Language Models:",
            fontsize=18,
            ha="center",
            color=COLOR_TEXT,
            alpha=min(1, progress * 4),
        )
        ax.text(
            5,
            3.0,
            "Attention Mechanism",
            fontsize=28,
            ha="center",
            color=COLOR_Q,
            weight="bold",
            alpha=min(1, max(0, progress * 4 - 1)),
        )
        ax.text(
            5,
            2.0,
            "(and its variants MHA, MQA, GQA)",
            fontsize=16,
            ha="center",
            color=COLOR_TEXT,
            alpha=min(1, max(0, progress * 4 - 2)),
        )
        ax.text(
            5,
            0.5,
            'Understanding how it helps models "focus" on important information',
            fontsize=12,
            ha="center",
            color="grey",
            alpha=min(1, max(0, progress * 2 - 0.5)),
        )

    # --- Scene 2: Basic Attention (Scaled Dot-Product) ---
    elif frame < SCENE_INTRO_FRAMES + SCENE_BASIC_ATTN_FRAMES:
        scene_frame = frame - SCENE_INTRO_FRAMES
        scene_duration = SCENE_BASIC_ATTN_FRAMES
        progress = scene_frame / scene_duration

        ax.text(
            5,
            5.5,
            "Basic Attention (Scaled Dot-Product)",
            fontsize=16,
            ha="center",
            color=COLOR_TEXT,
        )

        # Input Tokens (simplified)
        token_pos = [(2, 1), (4, 1), (6, 1), (8, 1)]
        token_labels = ["Token 1", "Token 2", "Token 3", "Token 4"]
        for i, (x, y) in enumerate(token_pos):
            draw_box(ax, x, y, 1.2, 0.6, token_labels[i], "lightgrey")

        # Q, K, V generation (focus on Token 2 generating Q)
        # 修改Q的位置，将其向上移动一些
        q_pos = (4, 2.8)  # 原来是(4, 2.5)
        # 修改K的位置，将它们分散开
        k_pos = [
            (x, 2.2) for x, y in token_pos
        ]  # 原来是[(x, 2.5) for x, y in token_pos]
        # 修改V的位置，将它们放在K的下方
        v_pos = [
            (x, 1.6) for x, y in token_pos
        ]  # 原来是[(x, 2.5) for x, y in token_pos]

        q_box = draw_box(
            ax, q_pos[0], q_pos[1], 0.8, 0.5, "Q", COLOR_Q, alpha=min(1, progress * 8)
        )
        draw_arrow(
            ax,
            (token_pos[1][0], token_pos[1][1] + 0.3),
            q_pos,
            alpha=min(1, progress * 8),
        )

        k_boxes = []
        if progress > 0.1:
            alpha_k = min(1, (progress - 0.1) * 8)
            for i, (x, y) in enumerate(k_pos):
                k_box = draw_box(ax, x, y, 0.8, 0.5, "K", COLOR_K, alpha=alpha_k)
                draw_arrow(
                    ax,
                    (token_pos[i][0], token_pos[i][1] + 0.3),
                    (x, y - 0.25),
                    alpha=alpha_k,
                )
                k_boxes.append(k_box)

        # Calculate Scores (Q dot K)
        # 修改score_pos的位置，使其更合理
        score_pos = [(x, 3.5) for x, y in k_pos]
        scores = np.array([0.8, 2.5, 1.2, 0.5])  # Example scores
        scaled_scores = scores / math.sqrt(4)  # Simulate scaling (d_k=4 hypothetical)
        softmax_scores = softmax(scaled_scores)

        if progress > 0.25:
            alpha_score = min(1, (progress - 0.25) * 5)
            ax.text(
                q_pos[0] - 1.8,  # 向左移动更多
                3.8,  # 调整垂直位置
                "Calculate Scores (Q·K / √dₖ):",
                ha="right",
                va="center",
                alpha=alpha_score,
                fontsize=10,
            )
            for i, (x, y) in enumerate(score_pos):
                draw_arrow(
                    ax, q_pos, (x, y - 0.1), color=COLOR_SCORES, alpha=alpha_score
                )
                draw_arrow(
                    ax, k_pos[i], (x, y - 0.1), color=COLOR_SCORES, alpha=alpha_score
                )
                ax.text(
                    x,
                    y,
                    f"{scaled_scores[i]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color=COLOR_SCORES,
                    alpha=alpha_score,
                )

        # Apply Softmax
        softmax_pos = [(x, 4.5) for x, y in k_pos]
        if progress > 0.5:
            alpha_softmax = min(1, (progress - 0.5) * 5)
            ax.text(
                q_pos[0] - 1.8,  # 向左移动更多
                4.8,  # 调整垂直位置
                "Softmax Normalization:",
                ha="right",
                va="center",
                alpha=alpha_softmax,
                fontsize=10,
            )
            for i, (x, y) in enumerate(softmax_pos):
                draw_arrow(
                    ax, score_pos[i], (x, y - 0.1), color="grey", alpha=alpha_softmax
                )
                ax.text(
                    x,
                    y,
                    f"{softmax_scores[i]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=10,
                    color=COLOR_SCORES,
                    weight="bold",
                    alpha=alpha_softmax,
                )

        # Weighted Sum of Values
        v_boxes = []
        # 修改output_pos的位置，避免与标题重叠
        output_pos = (q_pos[0], 5.0)  # 原来是(q_pos[0], 5.5)
        if progress > 0.7:
            alpha_v = min(1, (progress - 0.7) * 5)
            ax.text(
                q_pos[0] - 1.8,  # 向左移动更多
                5.3,  # 调整垂直位置，原来是5.8
                "Weighted Sum (Σ scoreᵢ * Vᵢ):",
                ha="right",
                va="center",
                alpha=alpha_v,
                fontsize=10,
            )
            for i, (x, y) in enumerate(v_pos):
                v_box = draw_box(
                    ax, x, y, 0.8, 0.5, "V", COLOR_V, alpha=alpha_v
                )  # 移除了alpha_v * 0.5，使V更明显
                v_boxes.append(v_box)
                draw_arrow(
                    ax,
                    (x, y + 0.25),  # 调整箭头起点
                    output_pos,
                    color=COLOR_V,
                    alpha=alpha_v * softmax_scores[i] * 2,
                )  # Arrow thickness/alpha by score
                draw_arrow(
                    ax, softmax_pos[i], output_pos, color=COLOR_SCORES, alpha=alpha_v
                )  # Arrow from score

            draw_box(
                ax,
                output_pos[0],
                output_pos[1],
                1.0,
                0.6,
                "Output",
                COLOR_OUTPUT,
                alpha=alpha_v,
            )

        # Formula
        if progress > 0.1:
            ax.text(
                5,
                0.3,
                r"$Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})V$",
                fontsize=14,
                ha="center",
                va="center",
                color=COLOR_TEXT,
                alpha=min(1, (progress - 0.1) * 3),
            )

    # --- Scene 3: Multi-Head Attention (MHA) ---
    elif frame < SCENE_INTRO_FRAMES + SCENE_BASIC_ATTN_FRAMES + SCENE_MHA_FRAMES:
        scene_frame = frame - (SCENE_INTRO_FRAMES + SCENE_BASIC_ATTN_FRAMES)
        scene_duration = SCENE_MHA_FRAMES
        progress = scene_frame / scene_duration

        ax.text(
            5,
            5.5,
            "Multi-Head Attention (MHA)",
            fontsize=16,
            ha="center",
            color=COLOR_TEXT,
        )
        ax.text(
            5,
            0.3,
            "Parallel computation of multiple Attention heads to capture different subspace information",
            fontsize=12,
            ha="center",
            color="grey",
        )

        # Input Q, K, V matrices
        qkv_y = 4.5
        draw_box(ax, 2, qkv_y, 1.5, 0.8, "Q", COLOR_Q, alpha=min(1, progress * 5))
        draw_box(ax, 5, qkv_y, 1.5, 0.8, "K", COLOR_K, alpha=min(1, progress * 5))
        draw_box(ax, 8, qkv_y, 1.5, 0.8, "V", COLOR_V, alpha=min(1, progress * 5))

        # Split into Heads
        head_colors = [COLOR_HEAD_1, COLOR_HEAD_2, COLOR_HEAD_3, COLOR_HEAD_4]
        num_heads = 4
        head_width = 1.5 / num_heads
        head_y = 3.5
        head_centers_q = [
            2 - 1.5 / 2 + head_width / 2 + i * head_width for i in range(num_heads)
        ]
        head_centers_k = [
            5 - 1.5 / 2 + head_width / 2 + i * head_width for i in range(num_heads)
        ]
        head_centers_v = [
            8 - 1.5 / 2 + head_width / 2 + i * head_width for i in range(num_heads)
        ]

        if progress > 0.1:
            alpha_split = min(1, (progress - 0.1) * 4)
            ax.text(
                5,
                head_y + 0.4,
                "1. Split into multiple 'Heads'",
                ha="center",
                va="center",
                alpha=alpha_split,
                fontsize=10,
            )
            for i in range(num_heads):
                # Q heads
                draw_box(
                    ax,
                    head_centers_q[i],
                    head_y,
                    head_width * 0.9,
                    0.6,
                    f"Q{i+1}",
                    head_colors[i],
                    alpha=alpha_split,
                )
                draw_arrow(
                    ax,
                    (head_centers_q[i], qkv_y - 0.4),
                    (head_centers_q[i], head_y + 0.3),
                    alpha=alpha_split * 0.7,
                )
                # K heads
                draw_box(
                    ax,
                    head_centers_k[i],
                    head_y,
                    head_width * 0.9,
                    0.6,
                    f"K{i+1}",
                    head_colors[i],
                    alpha=alpha_split,
                )
                draw_arrow(
                    ax,
                    (head_centers_k[i], qkv_y - 0.4),
                    (head_centers_k[i], head_y + 0.3),
                    alpha=alpha_split * 0.7,
                )
                # V heads
                draw_box(
                    ax,
                    head_centers_v[i],
                    head_y,
                    head_width * 0.9,
                    0.6,
                    f"V{i+1}",
                    head_colors[i],
                    alpha=alpha_split,
                )
                draw_arrow(
                    ax,
                    (head_centers_v[i], qkv_y - 0.4),
                    (head_centers_v[i], head_y + 0.3),
                    alpha=alpha_split * 0.7,
                )

        # Parallel Attention Calculation
        attn_out_y = 2.5
        attn_out_centers = [2, 4, 6, 8]
        if progress > 0.3:
            alpha_attn = min(1, (progress - 0.3) * 4)
            ax.text(
                5,
                attn_out_y + 0.4,
                "2. Parallel Attention Calculation",
                ha="center",
                va="center",
                alpha=alpha_attn,
                fontsize=10,
            )
            for i in range(num_heads):
                draw_box(
                    ax,
                    attn_out_centers[i],
                    attn_out_y,
                    1.0,
                    0.6,
                    f"Head {i+1}\nOutput",
                    head_colors[i],
                    alpha=alpha_attn,
                )
                # Simplified arrows showing calculation per head
                draw_arrow(
                    ax,
                    (head_centers_q[i], head_y - 0.3),
                    (attn_out_centers[i], attn_out_y + 0.3),
                    color=head_colors[i],
                    alpha=alpha_attn * 0.8,
                )
                draw_arrow(
                    ax,
                    (head_centers_k[i], head_y - 0.3),
                    (attn_out_centers[i], attn_out_y + 0.3),
                    color=head_colors[i],
                    alpha=alpha_attn * 0.8,
                )
                draw_arrow(
                    ax,
                    (head_centers_v[i], head_y - 0.3),
                    (attn_out_centers[i], attn_out_y + 0.3),
                    color=head_colors[i],
                    alpha=alpha_attn * 0.8,
                )

        # Concatenate and Project
        concat_y = 1.5
        final_out_y = 0.8
        if progress > 0.6:
            alpha_concat = min(1, (progress - 0.6) * 4)
            ax.text(
                5,
                concat_y + 0.3,
                "3. Concatenate",
                ha="center",
                va="center",
                alpha=alpha_concat,
                fontsize=10,
            )
            concat_box = patches.Rectangle(
                (5 - 1.5, concat_y - 0.3),
                3.0,
                0.6,
                linewidth=1,
                edgecolor="black",
                facecolor="none",
                alpha=alpha_concat,
            )
            ax.add_patch(concat_box)
            total_width = 0
            for i in range(num_heads):
                w = 1.0 * 0.9  # Width of head output used for concat vis
                draw_box(
                    ax,
                    5 - 1.5 + total_width + w / 2,
                    concat_y,
                    w,
                    0.6,
                    f"H{i+1}",
                    head_colors[i],
                    alpha=alpha_concat,
                )
                draw_arrow(
                    ax,
                    (attn_out_centers[i], attn_out_y - 0.3),
                    (5 - 1.5 + total_width + w / 2, concat_y + 0.3),
                    alpha=alpha_concat * 0.8,
                )
                total_width += w

        if progress > 0.8:
            alpha_proj = min(1, (progress - 0.8) * 5)
            ax.text(
                5,
                final_out_y + 0.3,
                "4. Final Linear Projection",
                ha="center",
                va="center",
                alpha=alpha_proj,
                fontsize=10,
            )
            draw_box(
                ax,
                5,
                final_out_y,
                1.5,
                0.6,
                "Final Output",
                COLOR_OUTPUT,
                alpha=alpha_proj,
            )
            draw_arrow(
                ax, (5, concat_y - 0.3), (5, final_out_y + 0.3), alpha=alpha_proj
            )

    # --- Scene 4: Multi-Query Attention (MQA) ---
    elif (
        frame
        < SCENE_INTRO_FRAMES
        + SCENE_BASIC_ATTN_FRAMES
        + SCENE_MHA_FRAMES
        + SCENE_MQA_FRAMES
    ):
        scene_frame = frame - (
            SCENE_INTRO_FRAMES + SCENE_BASIC_ATTN_FRAMES + SCENE_MHA_FRAMES
        )
        scene_duration = SCENE_MQA_FRAMES
        progress = scene_frame / scene_duration

        ax.text(
            5,
            5.5,
            "Multi-Query Attention (MQA)",
            fontsize=16,
            ha="center",
            color=COLOR_TEXT,
        )
        ax.text(
            5,
            0.3,
            "Multiple Query heads share a single set of Key/Value heads",
            fontsize=12,
            ha="center",
            color="grey",
        )
        ax.text(
            5,
            0.1,
            "Pros: Fast inference, less memory. Cons: Potential quality loss",
            fontsize=10,
            ha="center",
            color="grey",
        )

        # Multiple Q heads
        head_colors = [COLOR_HEAD_1, COLOR_HEAD_2, COLOR_HEAD_3, COLOR_HEAD_4]
        num_q_heads = 4
        q_head_y = 4.0
        q_head_centers = [2, 4, 6, 8]

        if progress > 0.05:
            alpha_q = min(1, progress * 8)
            ax.text(
                5,
                q_head_y + 0.5,
                "Multiple Query Heads",
                ha="center",
                va="center",
                alpha=alpha_q,
                fontsize=10,
            )
            for i in range(num_q_heads):
                draw_box(
                    ax,
                    q_head_centers[i],
                    q_head_y,
                    1.0,
                    0.6,
                    f"Q{i+1}",
                    head_colors[i],
                    alpha=alpha_q,
                )

        # Shared K and V heads
        kv_y = 2.5
        if progress > 0.2:
            alpha_kv = min(1, (progress - 0.2) * 5)
            ax.text(
                5,
                kv_y + 0.5,
                "Shared Key / Value Heads",
                ha="center",
                va="center",
                alpha=alpha_kv,
                fontsize=10,
            )
            draw_box(ax, 4, kv_y, 1.2, 0.7, "Shared K", COLOR_SHARED, alpha=alpha_kv)
            draw_box(ax, 6, kv_y, 1.2, 0.7, "Shared V", COLOR_SHARED, alpha=alpha_kv)

        # Attention Calculation (Arrows)
        output_y = 1.0
        if progress > 0.4:
            alpha_attn = min(1, (progress - 0.4) * 3)
            ax.text(
                5,
                output_y + 0.5,
                "Each Q Head calculates Attention with shared K/V",
                ha="center",
                va="center",
                alpha=alpha_attn,
                fontsize=10,
            )
            for i in range(num_q_heads):
                # Arrows from Q to shared K/V, then to a hypothetical output spot
                draw_arrow(
                    ax,
                    (q_head_centers[i], q_head_y - 0.3),
                    (4, kv_y + 0.35),
                    color=head_colors[i],
                    alpha=alpha_attn * 0.6,
                )  # Q to K
                draw_arrow(
                    ax,
                    (q_head_centers[i], q_head_y - 0.3),
                    (6, kv_y + 0.35),
                    color=head_colors[i],
                    alpha=alpha_attn * 0.6,
                )  # Q to V (conceptually)
                # Simplified: Show arrows converging towards a general output area
                draw_arrow(
                    ax,
                    (4, kv_y - 0.35),
                    (q_head_centers[i], output_y + 0.1),
                    color=COLOR_SHARED,
                    alpha=alpha_attn * 0.5,
                )  # K influence
                draw_arrow(
                    ax,
                    (6, kv_y - 0.35),
                    (q_head_centers[i], output_y + 0.1),
                    color=COLOR_SHARED,
                    alpha=alpha_attn * 0.5,
                )  # V influence
                # Draw small output box per head before final concat/proj (not shown)
                draw_box(
                    ax,
                    q_head_centers[i],
                    output_y - 0.2,
                    0.8,
                    0.4,
                    f"Out {i+1}",
                    head_colors[i],
                    alpha=alpha_attn * 0.8,
                )

    # --- Scene 5: Grouped-Query Attention (GQA) ---
    elif (
        frame
        < SCENE_INTRO_FRAMES
        + SCENE_BASIC_ATTN_FRAMES
        + SCENE_MHA_FRAMES
        + SCENE_MQA_FRAMES
        + SCENE_GQA_FRAMES
    ):
        scene_frame = frame - (
            SCENE_INTRO_FRAMES
            + SCENE_BASIC_ATTN_FRAMES
            + SCENE_MHA_FRAMES
            + SCENE_MQA_FRAMES
        )
        scene_duration = SCENE_GQA_FRAMES
        progress = scene_frame / scene_duration

        ax.text(
            5,
            5.5,
            "Grouped-Query Attention (GQA)",
            fontsize=16,
            ha="center",
            color=COLOR_TEXT,
        )
        ax.text(
            5,
            0.5,
            "Between MHA and MQA: Query heads in groups, sharing K/V within groups",
            fontsize=12,
            ha="center",
            color="grey",
        )
        ax.text(
            5,
            0.2,
            "Balancing performance and quality",
            fontsize=10,
            ha="center",
            color="grey",
        )

        # Multiple Q heads, grouped
        head_colors = [
            COLOR_HEAD_1,
            COLOR_HEAD_1,
            COLOR_HEAD_2,
            COLOR_HEAD_2,
        ]  # Example: 2 groups of 2
        group_colors = [COLOR_HEAD_1, COLOR_HEAD_2]
        num_q_heads = 4
        num_groups = 2
        q_heads_per_group = num_q_heads // num_groups

        q_head_y = 4.0
        q_head_centers = [1.5, 3.0, 6.0, 7.5]  # Spaced for groups

        if progress > 0.05:
            alpha_q = min(1, progress * 8)
            ax.text(
                2.25,
                q_head_y + 0.5,
                "Group 1 Q Heads",
                ha="center",
                va="center",
                alpha=alpha_q,
                fontsize=10,
                color=group_colors[0],
            )
            ax.text(
                6.75,
                q_head_y + 0.5,
                "Group 2 Q Heads",
                ha="center",
                va="center",
                alpha=alpha_q,
                fontsize=10,
                color=group_colors[1],
            )
            for i in range(num_q_heads):
                group_index = i // q_heads_per_group
                draw_box(
                    ax,
                    q_head_centers[i],
                    q_head_y,
                    1.0,
                    0.6,
                    f"Q{i+1}",
                    group_colors[group_index],
                    alpha=alpha_q,
                )

        # Shared K and V heads PER GROUP
        kv_y = 2.5
        kv_centers_k = [2.25, 6.75]  # One K per group
        kv_centers_v = [2.25, 6.75]  # One V per group (adjust x slightly for vis)

        if progress > 0.2:
            alpha_kv = min(1, (progress - 0.2) * 5)
            ax.text(
                5,
                kv_y + 0.7,  # 向上移动文本
                "Grouped Key / Value Heads within Groups",  # 修改文本
                ha="center",
                va="center",
                alpha=alpha_kv,
                fontsize=10,
            )
            for g in range(num_groups):
                # 调整K和V的位置，使它们更分开
                draw_box(
                    ax,
                    kv_centers_k[g] - 0.5,  # 向左移动
                    kv_y,
                    0.9,  # 减小宽度
                    0.7,
                    f"Group {g+1}\nK",  # 简化文本
                    COLOR_SHARED,
                    alpha=alpha_kv,
                )
                draw_box(
                    ax,
                    kv_centers_v[g] + 0.5,  # 向右移动
                    kv_y,
                    0.9,  # 减小宽度
                    0.7,
                    f"Group {g+1}\nV",  # 简化文本
                    COLOR_SHARED,
                    alpha=alpha_kv,
                )

        # Attention Calculation (Arrows within groups)
        output_y = 1.0
        if progress > 0.4:
            alpha_attn = min(1, (progress - 0.4) * 3)
            ax.text(
                5,
                output_y + 0.6,
                "Q Heads calculate Attention with K/V from their group",
                ha="center",
                va="center",
                alpha=alpha_attn,
                fontsize=10,
            )
            for i in range(num_q_heads):
                group_index = i // q_heads_per_group
                # Arrows from Q to its group's shared K/V
                draw_arrow(
                    ax,
                    (q_head_centers[i], q_head_y - 0.3),
                    (kv_centers_k[group_index] - 0.3, kv_y + 0.35),
                    color=group_colors[group_index],
                    alpha=alpha_attn * 0.6,
                )  # Q to K
                draw_arrow(
                    ax,
                    (q_head_centers[i], q_head_y - 0.3),
                    (kv_centers_v[group_index] + 0.3, kv_y + 0.35),
                    color=group_colors[group_index],
                    alpha=alpha_attn * 0.6,
                )  # Q to V (conceptually)
                # Simplified arrows from shared K/V to output area for that Q
                draw_arrow(
                    ax,
                    (kv_centers_k[group_index] - 0.3, kv_y - 0.35),
                    (q_head_centers[i], output_y + 0.2),
                    color=COLOR_SHARED,
                    alpha=alpha_attn * 0.5,
                )  # K influence
                draw_arrow(
                    ax,
                    (kv_centers_v[group_index] + 0.3, kv_y - 0.35),
                    (q_head_centers[i], output_y + 0.2),
                    color=COLOR_SHARED,
                    alpha=alpha_attn * 0.5,
                )  # V influence
                # Draw small output box per head
                draw_box(
                    ax,
                    q_head_centers[i],
                    output_y - 0.1,
                    0.8,
                    0.4,
                    f"Out {i+1}",
                    group_colors[group_index],
                    alpha=alpha_attn * 0.8,
                )

    # --- Scene 6: Conclusion (Outro) ---
    else:  # Frames >= TOTAL_FRAMES - SCENE_OUTRO_FRAMES
        scene_frame = frame - (TOTAL_FRAMES - SCENE_OUTRO_FRAMES)
        scene_duration = SCENE_OUTRO_FRAMES
        progress = scene_frame / scene_duration

        ax.text(
            5,
            5.0,
            "Summary",
            fontsize=20,
            ha="center",
            color=COLOR_TEXT,
            alpha=min(1, progress * 4),
        )
        ax.text(
            5,
            4.0,
            "Attention is key for LLMs to understand context",
            fontsize=14,
            ha="center",
            color=COLOR_TEXT,
            alpha=min(1, max(0, progress * 4 - 0.5)),
        )
        ax.text(
            5,
            3.0,
            "MHA: Captures diverse information, computationally intensive",
            fontsize=14,
            ha="center",
            color=COLOR_HEAD_1,
            alpha=min(1, max(0, progress * 4 - 1.0)),
        )
        ax.text(
            5,
            2.5,
            "MQA: Fast inference, memory efficient, potential accuracy trade-off",
            fontsize=14,
            ha="center",
            color=COLOR_SHARED,
            alpha=min(1, max(0, progress * 4 - 1.5)),
        )
        ax.text(
            5,
            2.0,
            "GQA: Compromise between MHA and MQA",
            fontsize=14,
            ha="center",
            color=COLOR_HEAD_2,
            alpha=min(1, max(0, progress * 4 - 2.0)),
        )
        ax.text(
            5,
            1.0,
            "Understanding these variants helps choose appropriate model architectures",
            fontsize=12,
            ha="center",
            color="grey",
            alpha=min(1, max(0, progress * 4 - 2.5)),
        )

    # Add frame number for debugging if needed
    # ax.text(9.8, 0.2, f"Frame: {frame}", ha='right', va='bottom', fontsize=8, color='grey')


# --- Create and Save Animation ---
print("Creating animation...")
# Increase interval slightly if rendering feels rushed, decrease for faster animation (but may need higher FPS)
ani = animation.FuncAnimation(
    fig, update, frames=TOTAL_FRAMES, interval=1000 / FPS, blit=False
)

try:
    # Save the animation
    output_filename = "attention_mechanism_explained——3.mp4"
    print(f"Saving as {output_filename} (this may take a few minutes)...")
    start_time = time.time()
    ani.save(
        output_filename, writer="ffmpeg", fps=FPS, dpi=150
    )  # Adjust dpi for resolution
    end_time = time.time()
    print(
        f"Animation saved successfully! Time taken: {end_time - start_time:.2f} seconds"
    )

    # Optionally display the animation plot window (remove if running in a non-GUI environment)
    # print("Displaying animation preview (if environment supports)...")
    # plt.show()

except FileNotFoundError:
    print("\nError: FFmpeg not found.")
    print("Please make sure FFmpeg is installed and added to your system PATH.")
    print("Visit https://ffmpeg.org/download.html to download.")
except Exception as e:
    print(f"\nError creating or saving animation: {e}")
    print("Please check your matplotlib, numpy, and FFmpeg installation.")
