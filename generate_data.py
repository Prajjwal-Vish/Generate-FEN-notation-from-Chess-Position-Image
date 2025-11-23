#!/usr/bin/env python3
"""
generate_dataset.py

Improved synthetic chess tile dataset generator.

Features:
 - Multiple real piece themes (uses chess.svg.style.Style when available)
 - Board color variations
 - Optional full-board PNG saving
 - Simple augmentations (brightness, Gaussian blur)
 - Saves tiles into organized folders under OUTPUT_DIR
"""

import os
import random
from pathlib import Path
import io
import sys

import chess
import chess.svg
import cv2
import numpy as np

# SVG -> raster conversion
# Make sure svglib and reportlab are installed in the venv:
# pip install svglib reportlab
try:
    from svglib.svglib import svg2rlg
    from reportlab.graphics import renderPM
except Exception as e:
    print("ERROR: svglib/reportlab required to convert SVG->PNG.")
    print("Install with: pip install svglib reportlab")
    raise

# Try to import the Style enum from python-chess (some versions expose it)
USE_PYCHESS_STYLES = False
AVAILABLE_PYCHESS_STYLES = []
try:
    import chess.svg.style as chess_svg_style
    # chess_svg_style.Style is expected; inspect available attributes
    if hasattr(chess_svg_style, "Style"):
        # collect a few known style names if present (safe-guard)
        for name in dir(chess_svg_style.Style):
            if name.isupper():
                AVAILABLE_PYCHESS_STYLES.append(getattr(chess_svg_style.Style, name))
        if AVAILABLE_PYCHESS_STYLES:
            USE_PYCHESS_STYLES = True
except Exception:
    # We'll fallback to CSS-injection below
    USE_PYCHESS_STYLES = False

# ---------- Configuration ----------
NUM_BOARDS_TO_GENERATE = 1000      # number of boards to generate
NUM_RANDOM_MOVES = 30              # number of random moves applied to each board
OUTPUT_DIR = "synthetic_data2"      # base output directory
SVG_BOARD_SIZE = 390               # size in px of generated SVG board (multiple of 8 recommended)
SQUARE_SIZE = SVG_BOARD_SIZE // 8
SAVE_FULL_BOARD_PNG = True         # also save full board PNGs (in OUTPUT_DIR/full_boards/)
APPLY_AUGMENTATIONS = True         # apply simple augmentations (brightness + blur)
SAVE_BOARDS_EVERY_N = 100          # progress message frequency
# -----------------------------------

# Piece folder mapping (same as your previous mapping)
PIECE_TO_DIR_MAP = {
    'p': 'dark_pawn', 'r': 'dark_rook', 'n': 'dark_knight', 'b': 'dark_bishop',
    'q': 'dark_queen', 'k': 'dark_king',
    'P': 'light_pawn', 'R': 'light_rook', 'N': 'light_knight', 'B': 'light_bishop',
    'Q': 'light_queen', 'K': 'light_king'
}

# Board color variations (pairs: light square color, dark square color)
BOARD_COLOR_THEMES = [
    ("#f0d9b5", "#b58863"),  # classic wood
    ("#f8f8f8", "#888888"),  # grey / neutral
    ("#eaf6df", "#4e8b3a"),  # green-ish
    ("#fff7e6", "#a86b2d"),  # warm wood
    ("#f3f6ff", "#556bff"),  # blue-ish contrast
]

# Fallback CSS styles (only alter piece colors / stroke -- cannot change piece shape)
FALLBACK_CSS_STYLES = {
    "merida_css": """
    <style>
      .piece.white { fill: #ffffff !important; }
      .piece.black { fill: #111111 !important; }
    </style>
    """,
    "alpha_css": """
    <style>
      .piece.white { fill: #f1f1f1 !important; stroke: #333 !important; }
      .piece.black { fill: #202020 !important; stroke: #000 !important; }
    </style>
    """,
    "cburnett_css": """
    <style>
      .piece.white { fill: #fffdf5 !important; stroke: #222 !important; }
      .piece.black { fill: #111 !important; stroke: #eee !important; }
    </style>
    """,
}
FALLBACK_STYLE_NAMES = list(FALLBACK_CSS_STYLES.keys())

# If python-chess provided styles are available, map them (name -> style object)
PYCHESS_STYLE_NAMES = []
if USE_PYCHESS_STYLES:
    # Build a small list of commonly available style enums (if present)
    # We don't fail if some of these names are not there; we check attributes.
    candidate_names = ["MERIDA", "CBURNETT", "CHEQMERICA", "ALPHA", "STAUNTY", "USCF"]
    for nm in candidate_names:
        if hasattr(chess_svg_style.Style, nm):
            obj = getattr(chess_svg_style.Style, nm)
            PYCHESS_STYLE_NAMES.append((nm.lower(), obj))

if USE_PYCHESS_STYLES:
    print("python-chess style support detected. Will use real piece themes:", [n for n, _ in PYCHESS_STYLE_NAMES])
else:
    print("python-chess style support NOT detected. Falling back to CSS color modifications only.")

# ---------- Utility functions ----------

def create_dirs(base_out: str):
    """Create folders for each piece type and empty squares."""
    base = Path(base_out)
    base.mkdir(parents=True, exist_ok=True)

    # piece dirs
    for d in PIECE_TO_DIR_MAP.values():
        (base / d).mkdir(exist_ok=True)

    # empty squares
    (base / "empty_dark").mkdir(exist_ok=True)
    (base / "empty_light").mkdir(exist_ok=True)

    # optional: full boards
    if SAVE_FULL_BOARD_PNG:
        (base / "full_boards").mkdir(exist_ok=True)

def get_piece_at(fen_rank: str, file_index: int) -> str:
    """Return the piece character at file_index for a single FEN rank string."""
    current_file = 0
    for ch in fen_rank:
        if ch.isdigit():
            current_file += int(ch)
        else:
            if current_file == file_index:
                return ch
            current_file += 1
    return '1'

def svg_to_png_bytes(svg_string: str) -> bytes:
    """Convert SVG string to PNG bytes using svglib + reportlab."""
    svg_file_like = io.StringIO(svg_string)
    drawing = svg2rlg(svg_file_like)
    png_bytes = renderPM.drawToString(drawing, fmt="PNG")
    return png_bytes

def apply_augmentations(img: np.ndarray) -> np.ndarray:
    """Apply simple random augmentations to an OpenCV image (BGR)."""
    out = img.copy()

    # Random brightness (scale)
    if random.random() < 0.7:
        factor = random.uniform(0.85, 1.25)
        out = np.clip(out.astype(np.float32) * factor, 0, 255).astype(np.uint8)

    # Random gaussian blur
    if random.random() < 0.35:
        k = random.choice([1, 3, 5])  # kernel sizes (1=none)
        if k > 1:
            out = cv2.GaussianBlur(out, (k, k), 0)

    return out

# ---------- Main generation loop ----------
def generate():
    print("Creating directories...")
    create_dirs(OUTPUT_DIR)

    total_expected_tiles = NUM_BOARDS_TO_GENERATE * 64
    print(f"Generating {NUM_BOARDS_TO_GENERATE} boards -> ~{total_expected_tiles} tiles (will save to '{OUTPUT_DIR}')")

    for i in range(NUM_BOARDS_TO_GENERATE):
        board = chess.Board()
        # randomize game state by making random legal moves
        try:
            for _ in range(NUM_RANDOM_MOVES):
                if board.is_game_over():
                    break
                board.push(random.choice(list(board.legal_moves)))
        except Exception:
            # fallback: if something weird happens, skip this board
            continue

        fen_str = board.fen().split(" ")[0]  # only piece placement
        fen_ranks = fen_str.split("/")

        # Choose style:
        use_pychess_style = USE_PYCHESS_STYLES and bool(PYCHESS_STYLE_NAMES) and (random.random() < 0.8)
        if use_pychess_style:
            style_name, style_obj = random.choice(PYCHESS_STYLE_NAMES)
            # board square colors too
            light_color, dark_color = random.choice(BOARD_COLOR_THEMES)
            svg_string = chess.svg.board(board=board, size=SVG_BOARD_SIZE, style=style_obj,
                                         colors={"square light": light_color, "square dark": dark_color})
            chosen_style_desc = f"pychess:{style_name}"
        else:
            # fallback: default shapes but with CSS color modifications + board color pair
            css_style_name = random.choice(FALLBACK_STYLE_NAMES)
            css_block = FALLBACK_CSS_STYLES[css_style_name]
            light_color, dark_color = random.choice(BOARD_COLOR_THEMES)
            svg_string = chess.svg.board(board=board, size=SVG_BOARD_SIZE,
                                         colors={"square light": light_color, "square dark": dark_color})
            # inject CSS before closing tag
            svg_string = svg_string.replace("</svg>", css_block + "</svg>")
            chosen_style_desc = f"css:{css_style_name}"

        # convert SVG to PNG bytes
        try:
            png_bytes = svg_to_png_bytes(svg_string)
        except Exception as e:
            print(f"[WARN] SVG->PNG conversion failed for board {i}: {e}")
            continue

        # decode PNG to cv2 image (BGR)
        nparr = np.frombuffer(png_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            print(f"[WARN] Could not decode PNG for board {i}")
            continue

        # optionally save full board
        if SAVE_FULL_BOARD_PNG:
            full_name = Path(OUTPUT_DIR) / "full_boards" / f"board_{i}_style_{chosen_style_desc}.png"
            # maybe apply augmentation to full board as well
            full_img_to_save = apply_augmentations(img) if APPLY_AUGMENTATIONS else img
            cv2.imwrite(str(full_name), full_img_to_save)

        # slice tiles and save
        for r in range(8):
            for c in range(8):
                piece_char = get_piece_at(fen_ranks[r], c)
                if piece_char == '1':
                    save_folder = "empty_light" if (r + c) % 2 == 0 else "empty_dark"
                else:
                    save_folder = PIECE_TO_DIR_MAP.get(piece_char)
                if not save_folder:
                    # unknown symbol; skip
                    continue

                y1, y2 = r * SQUARE_SIZE, (r + 1) * SQUARE_SIZE
                x1, x2 = c * SQUARE_SIZE, (c + 1) * SQUARE_SIZE
                square_img = img[y1:y2, x1:x2]

                if square_img.shape[0] == 0 or square_img.shape[1] == 0:
                    # defensive check: skip invalid crops
                    continue

                # augment per-tile sometimes
                if APPLY_AUGMENTATIONS and random.random() < 0.5:
                    square_img = apply_augmentations(square_img)

                img_name = f"board_{i}_style_{chosen_style_desc}_rank_{r}_file_{c}.png"
                save_path = Path(OUTPUT_DIR) / save_folder / img_name
                cv2.imwrite(str(save_path), square_img)

        if (i + 1) % SAVE_BOARDS_EVERY_N == 0:
            print(f"  ...generated {i+1}/{NUM_BOARDS_TO_GENERATE} boards")

    print("Generation complete.")
    print(f"Look inside '{OUTPUT_DIR}' for folders (e.g., {', '.join(list(PIECE_TO_DIR_MAP.values())[:3])}, ... , empty_light).")
    if SAVE_FULL_BOARD_PNG:
        print(f"Full boards saved in: {OUTPUT_DIR}/full_boards/")

if __name__ == "__main__":
    generate()
