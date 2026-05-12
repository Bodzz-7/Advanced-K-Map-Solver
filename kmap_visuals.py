"""
kmap_visuals.py

Tkinter Canvas widget for drawing Karnaugh Maps (K-Maps) for 2, 3, or 4 variables.

Supports:
- Drawing a labeled K-map grid with Gray code headers (00, 01, 11, 10).
- Populating cell values via update_map(minterms, dont_cares).
- (Stretch) Drawing semi-transparent colored overlays for implicants via draw_loops(prime_implicants).

Variable conventions used in this widget:
- 2 variables: rows = A, columns = B
- 3 variables: rows = A, columns = B C
- 4 variables: rows = A B, columns = C D

This is a standard layout for K-map visualization.
"""

from __future__ import annotations

import itertools
import tkinter as tk
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple


def _gray_codes(bits: int) -> List[str]:
    """
    Return Gray code sequence strings for a given bit width.

    Examples:
        bits=1 -> ['0', '1']
        bits=2 -> ['00', '01', '11', '10']
    """

    if bits <= 0:
        return [""]
    if bits == 1:
        return ["0", "1"]
    if bits == 2:
        return ["00", "01", "11", "10"]

    # General (not strictly needed for 2..4 vars, but included for completeness):
    prev = _gray_codes(bits - 1)
    return ["0" + s for s in prev] + ["1" + s for s in reversed(prev)]


class KMapCanvas(tk.Canvas):
    """
    Canvas-based Karnaugh Map grid for 2/3/4 variables.
    """

    def __init__(
        self,
        parent: tk.Widget,
        num_variables: int = 4,
        *,
        cell_size: int = 56,
        padding: int = 16,
        header_size: int = 34,
        bg: str = "#1e1e1e",
        grid_bg: str = "#2d2d2d",
        grid_line: str = "#505050",
        text_color: str = "#ffffff",
        header_bg: str = "#252525",
        header_text: str = "#ffffff",
        **kwargs,
    ) -> None:
        super().__init__(parent, bg=bg, highlightthickness=0, **kwargs)

        self.num_variables = num_variables
        self.cell_size = cell_size
        self.padding = padding
        # Backwards-compatible attribute: older versions used a separate header strip size.
        # The grid logic below now uses the "top-left cell" approach where headers are
        # just regular cells with the same size as data cells.
        self.header_size = header_size

        self.colors = {
            "bg": bg,
            "grid_bg": grid_bg,
            "grid_line": grid_line,
            "text": text_color,
            "header_bg": header_bg,
            "header_text": header_text,
        }

        # Internal caches for fast updates:
        # - map row/col index -> cell bounding box (x1,y1,x2,y2)
        self._cell_boxes: Dict[Tuple[int, int], Tuple[int, int, int, int]] = {}
        # - map row/col index -> embedded button widget
        self._cell_buttons: Dict[Tuple[int, int], tk.Button] = {}
        # - map row/col index -> canvas window id (for cleanup)
        self._cell_window_ids: Dict[Tuple[int, int], int] = {}
        # - map minterm index -> (row, col) for this layout
        self._minterm_to_rc: Dict[int, Tuple[int, int]] = {}

        # Overlay items for loops (to clear/redraw easily)
        self._loop_item_ids: List[int] = []

        # Callback invoked when user clicks a K-map data cell button.
        # Signature: callback(row, col)
        self._cell_click_callback: Optional[Callable[[int, int], None]] = None

        self.draw_grid(num_variables)

    # -----------------------------
    # Public API
    # -----------------------------

    def set_cell_click_callback(self, callback: Optional[Callable[[int, int], None]]) -> None:
        """
        Register a callback invoked when any K-map data cell is clicked.
        The callback receives (row, col) coordinates of the data cell (0-based within the map).
        """

        self._cell_click_callback = callback

    def draw_grid(self, num_variables: int) -> None:
        """
        Clear and redraw the K-map grid for 2, 3, or 4 variables.
        """

        if num_variables not in (2, 3, 4):
            raise ValueError("KMapCanvas supports only 2, 3, or 4 variables.")

        self.num_variables = num_variables
        self.delete("all")
        self._cell_boxes.clear()
        for btn in self._cell_buttons.values():
            try:
                btn.destroy()
            except tk.TclError:
                pass
        self._cell_buttons.clear()
        self._cell_window_ids.clear()
        self._minterm_to_rc.clear()
        self._loop_item_ids.clear()

        row_bits, col_bits = self._row_col_bits(num_variables)
        row_gray = _gray_codes(row_bits)
        col_gray = _gray_codes(col_bits)

        rows = len(row_gray)
        cols = len(col_gray)

        # New header layout (to match Tkinter "grid rules" concept):
        #
        # Determine labels:
        # - row_label: "A" / "A" / "AB"
        # - col_label: "B" / "BC" / "CD"
        #
        # Placement (conceptual grid of cells):
        # - Row 0: col_label at (0,1) spanning across all data columns (columnspan = cols)
        # - Row 1: row_label at (1,0); column Gray headers at (1,1..cols)
        # - Row 2..: row Gray headers at (2..,0); data cells at (2..,1..)
        #
        # Therefore total cell grid size is:
        #   rows_total = rows + 2
        #   cols_total = cols + 1
        width = self.padding * 2 + (cols + 1) * self.cell_size
        height = self.padding * 2 + (rows + 2) * self.cell_size
        self.configure(width=width, height=height, scrollregion=(0, 0, width, height))

        x0 = self.padding
        y0 = self.padding

        row_label, col_label = self._axis_labels(num_variables)

        # Typography hierarchy:
        # - col_label (top spanning) is the primary header
        # - row_label and Gray codes are secondary headers
        # - cell values remain the most prominent inside the map
        font_primary = ("Segoe UI", 11, "bold")
        font_header = ("Segoe UI", 10, "bold")
        font_gray = ("Segoe UI", 9, "bold")
        font_cell = ("Segoe UI", 12, "bold")

        # (0,0) empty corner cell (keeps the table look consistent)
        self.create_rectangle(
            x0,
            y0,
            x0 + self.cell_size,
            y0 + self.cell_size,
            fill=self.colors["header_bg"],
            outline=self.colors["grid_line"],
            width=1,
        )

        # Row 0: spanning col_label starting at column=1 with "columnspan = cols"
        span_x1 = x0 + 1 * self.cell_size
        span_y1 = y0
        span_x2 = span_x1 + cols * self.cell_size
        span_y2 = span_y1 + self.cell_size
        self.create_rectangle(
            span_x1,
            span_y1,
            span_x2,
            span_y2,
            fill=self.colors["header_bg"],
            outline=self.colors["grid_line"],
            width=1,
        )
        self.create_text(
            (span_x1 + span_x2) // 2,
            (span_y1 + span_y2) // 2,
            text=col_label,
            fill=self.colors["header_text"],
            font=font_primary,
        )

        # Row 1, column 0: row_label
        rlab_x1 = x0
        rlab_y1 = y0 + 1 * self.cell_size
        rlab_x2 = rlab_x1 + self.cell_size
        rlab_y2 = rlab_y1 + self.cell_size
        self.create_rectangle(
            rlab_x1,
            rlab_y1,
            rlab_x2,
            rlab_y2,
            fill=self.colors["header_bg"],
            outline=self.colors["grid_line"],
            width=1,
        )
        self.create_text(
            (rlab_x1 + rlab_x2) // 2,
            (rlab_y1 + rlab_y2) // 2,
            text=row_label,
            fill=self.colors["header_text"],
            font=font_header,
        )

        # Row 1: column Gray headers starting at column=1
        for c, code in enumerate(col_gray):
            hx1 = x0 + (c + 1) * self.cell_size
            hy1 = y0 + 1 * self.cell_size
            hx2 = hx1 + self.cell_size
            hy2 = hy1 + self.cell_size
            self.create_rectangle(
                hx1,
                hy1,
                hx2,
                hy2,
                fill=self.colors["header_bg"],
                outline=self.colors["grid_line"],
                width=1,
            )
            self.create_text(
                (hx1 + hx2) // 2,
                (hy1 + hy2) // 2,
                text=code,
                fill=self.colors["header_text"],
                font=font_gray,
            )

        # Column 0: row Gray headers starting at row=2
        for r, code in enumerate(row_gray):
            hx1 = x0
            hy1 = y0 + (r + 2) * self.cell_size
            hx2 = hx1 + self.cell_size
            hy2 = hy1 + self.cell_size
            self.create_rectangle(
                hx1,
                hy1,
                hx2,
                hy2,
                fill=self.colors["header_bg"],
                outline=self.colors["grid_line"],
                width=1,
            )
            self.create_text(
                (hx1 + hx2) // 2,
                (hy1 + hy2) // 2,
                text=code,
                fill=self.colors["header_text"],
                font=font_gray,
            )

        # Data cells start at row=2, col=1 (shifted down by one more row).
        for r in range(rows):
            for c in range(cols):
                x1 = x0 + (c + 1) * self.cell_size
                y1 = y0 + (r + 2) * self.cell_size
                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size
                self._cell_boxes[(r, c)] = (x1, y1, x2, y2)

                self.create_rectangle(
                    x1,
                    y1,
                    x2,
                    y2,
                    fill=self.colors["grid_bg"],
                    outline=self.colors["grid_line"],
                    width=1,
                )

                
                btn = tk.Button(
                    self,
                    text="0",
                    font=font_cell,
                    fg=self.colors["text"],
                    bg=self.colors["grid_bg"],
                    activebackground="#3a3a3a",
                    activeforeground=self.colors["text"],
                    relief="flat",
                    bd=0,
                    highlightthickness=0,
                    cursor="hand2",
                    command=lambda rr=r, cc=c: self._handle_cell_click(rr, cc),
                )
                win_id = self.create_window(
                    (x1 + x2) // 2,
                    (y1 + y2) // 2,
                    window=btn,
                    width=self.cell_size - 10,
                    height=self.cell_size - 10,
                )
                self._cell_buttons[(r, c)] = btn
                self._cell_window_ids[(r, c)] = win_id

     
        self._minterm_to_rc = self._build_minterm_mapping(num_variables, row_gray, col_gray)

    def set_cell_value(self, row: int, col: int, value: str) -> None:
        """
        Update a single K-map data cell display.
        Value must be one of: '0', '1', 'X'
        """

        btn = self._cell_buttons.get((row, col))
        if not btn:
            return
        if value not in ("0", "1", "X"):
            return

        # Color-coding:
        if value == "1":
            fg = "#34d399"  # green
        elif value == "X":
            fg = "#fbbf24"  # amber
        else:
            fg = self.colors["text"]

        btn.configure(text=value, fg=fg)

    def update_map(self, minterms: List[int], dont_cares: List[int]) -> None:
        """
        Populate grid values:
        - '1' for indices in minterms
        - 'X' for indices in dont_cares
        - '0' otherwise
        """

        n = self.num_variables
        max_index = (1 << n) - 1

        m_set = set(minterms or [])
        d_set = set(dont_cares or [])

        overlap = m_set & d_set
        if overlap:
            raise ValueError(f"Overlap between minterms and dont_cares: {sorted(overlap)}")

        for v in sorted(m_set | d_set):
            if not isinstance(v, int):
                raise TypeError("minterms and dont_cares must contain integers.")
            if v < 0 or v > max_index:
                raise ValueError(f"Index {v} out of range for {n} variables (0..{max_index}).")

        
        self._clear_loops()

        # Set all cells to 0 first.
        for (r, c) in self._cell_buttons.keys():
            self.set_cell_value(r, c, "0")

        # Apply X and 1. (Order matters: '1' should override 'X' if caller misuses lists.)
        for v in d_set:
            rc = self._minterm_to_rc.get(v)
            if rc is None:
                continue
            self.set_cell_value(rc[0], rc[1], "X")

        for v in m_set:
            rc = self._minterm_to_rc.get(v)
            if rc is None:
                continue
            self.set_cell_value(rc[0], rc[1], "1")

    def draw_loops(self, prime_implicants: List[str]) -> None:
        """
        Visualize implicants as overlays on the map.
        Supports wrap-around drawing for K-Map edges.
        """

        self._clear_loops()

        n = self.num_variables
        palette = ["#60a5fa", "#f472b6", "#a78bfa", "#fb7185", "#22c55e", "#fbbf24"]

        row_bits, col_bits = self._row_col_bits(n)
        num_rows = 1 << row_bits
        num_cols = 1 << col_bits

        for idx, pat in enumerate(prime_implicants or []):
            if not isinstance(pat, str) or len(pat) != n:
                continue
            if any(ch not in ("0", "1", "-") for ch in pat):
                continue

            color = palette[idx % len(palette)]
            covered_indices = self._expand_pattern_to_minterms(pat, n)

            rcs = []
            for m in covered_indices:
                rc = self._minterm_to_rc.get(m)
                if rc is not None and rc in self._cell_boxes:
                    rcs.append(rc)
                    
            if not rcs:
                continue
                
            rows_covered = sorted(list(set(r for r, c in rcs)))
            cols_covered = sorted(list(set(c for r, c in rcs)))
            
            def get_segments(indices: List[int]) -> List[List[int]]:
                if not indices: return []
                segs = []
                curr = [indices[0]]
                for i in range(1, len(indices)):
                    if indices[i] == curr[-1] + 1:
                        curr.append(indices[i])
                    else:
                        segs.append(curr)
                        curr = [indices[i]]
                segs.append(curr)
                return segs
                
            row_segs = get_segments(rows_covered)
            col_segs = get_segments(cols_covered)
            
            pad = 2  # Inner padding for the loop
            
            for r_seg in row_segs:
                for c_seg in col_segs:
                    r_min, r_max = r_seg[0], r_seg[-1]
                    c_min, c_max = c_seg[0], c_seg[-1]
                    
                    x1, y1, _, _ = self._cell_boxes[(r_min, c_min)]
                    _, _, x2, y2 = self._cell_boxes[(r_max, c_max)]
                    
                    px1, py1 = x1 + pad, y1 + pad
                    px2, py2 = x2 - pad, y2 - pad
                    
                    wraps_top = r_min == 0 and len(row_segs) > 1
                    wraps_bot = r_max == num_rows - 1 and len(row_segs) > 1
                    wraps_left = c_min == 0 and len(col_segs) > 1
                    wraps_right = c_max == num_cols - 1 and len(col_segs) > 1
                    
                    if wraps_top: py1 = y1
                    if wraps_bot: py2 = y2
                    if wraps_left: px1 = x1
                    if wraps_right: px2 = x2
                    
                    rect_id = self.create_rectangle(
                        px1, py1, px2, py2,
                        fill=color, outline="", stipple="gray25"
                    )
                    self._loop_item_ids.append(rect_id)
                    
                    # Draw thick borders to emphasize the loop
                    b_kwargs = {"fill": color, "width": 3}
                    if not wraps_top:
                        l = self.create_line(px1, py1, px2, py1, **b_kwargs)
                        self._loop_item_ids.append(l)
                    if not wraps_bot:
                        l = self.create_line(px1, py2, px2, py2, **b_kwargs)
                        self._loop_item_ids.append(l)
                    if not wraps_left:
                        l = self.create_line(px1, py1, px1, py2, **b_kwargs)
                        self._loop_item_ids.append(l)
                    if not wraps_right:
                        l = self.create_line(px2, py1, px2, py2, **b_kwargs)
                        self._loop_item_ids.append(l)

    def _handle_cell_click(self, row: int, col: int) -> None:
        if self._cell_click_callback is not None:
            self._cell_click_callback(row, col)

    # -----------------------------
    # Internals
    # -----------------------------

    @staticmethod
    def _row_col_bits(num_variables: int) -> Tuple[int, int]:
        if num_variables == 2:
            return (1, 1)
        if num_variables == 3:
            return (1, 2)
        if num_variables == 4:
            return (2, 2)
        raise ValueError("KMapCanvas supports only 2, 3, or 4 variables.")

    @staticmethod
    def _axis_labels(num_variables: int) -> Tuple[str, str]:
        """
        Return (row_label, col_label) indicating which variables are mapped to each axis.
        """

        if num_variables == 2:
            return ("A", "B")
        if num_variables == 3:
            return ("A", "BC")
        if num_variables == 4:
            return ("AB", "CD")
        return ("", "")

    @staticmethod
    def _build_minterm_mapping(
        num_variables: int, row_gray: Sequence[str], col_gray: Sequence[str]
    ) -> Dict[int, Tuple[int, int]]:
        """
        Build mapping: minterm index -> (row, col), based on the variable convention.

        Convention recap:
        - 2 vars: row bits = A, col bits = B
        - 3 vars: row bits = A, col bits = B C
        - 4 vars: row bits = A B, col bits = C D
        """

        row_bits, col_bits = KMapCanvas._row_col_bits(num_variables)
        n = num_variables

        row_pos = {code: i for i, code in enumerate(row_gray)}
        col_pos = {code: i for i, code in enumerate(col_gray)}

        mapping: Dict[int, Tuple[int, int]] = {}
        for m in range(1 << n):
            bits = format(m, f"0{n}b")  # e.g., ABCD
            r_bits = bits[:row_bits]
            c_bits = bits[row_bits : row_bits + col_bits]
            r = row_pos[r_bits]
            c = col_pos[c_bits]
            mapping[m] = (r, c)
        return mapping

    def _expand_pattern_to_minterms(self, pattern: str, n: int) -> Set[int]:
        """
        Expand a pattern with '-' into the concrete minterm indices it covers.

        Example for n=3:
            pattern = '1-0' -> covers 100 (4) and 110 (6)
        """

        choices: List[Sequence[str]] = []
        for ch in pattern:
            if ch == "-":
                choices.append(("0", "1"))
            else:
                choices.append((ch,))

        out: Set[int] = set()
        for bits in itertools.product(*choices):
            out.add(int("".join(bits), 2))
        return out

    def _clear_loops(self) -> None:
        for item_id in self._loop_item_ids:
            try:
                self.delete(item_id)
            except tk.TclError:
                pass
        self._loop_item_ids.clear()

