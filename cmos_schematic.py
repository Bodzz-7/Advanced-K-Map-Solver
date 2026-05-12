"""
cmos_schematic.py

CMOS Transistor Schematic Visualizer
-------------------------------------

Given a minimized SOP Boolean expression (from qm_algorithm.py), this module:

1. Parses the SOP string into a structured list of product terms.
2. Factors shared literals out of product terms using a greedy recursive
   algorithm, producing a compact NetworkNode tree with no duplicate transistors.
3. Derives the dual PUN (PMOS) topology from the factored PDN tree.
4. Renders both networks on a Tkinter Canvas with:
   - VDD rail at top
   - PUN (PMOS) below VDD
   - Output node (F) in the middle
   - PDN (NMOS) below F
   - GND rail at bottom
   - Standard MOSFET gate symbols
   - Labeled gate inputs (variable names with complement marks)

Transistor sharing  (key improvement over the naive build)
----------------------------------------------------------
A naive SOP -> PDN mapping places one transistor per literal per term, so
shared literals appear multiple times.  Example:

  Naive   AB' + B'C  ->  parallel( series(A,B'),  series(B',C) )   4 transistors

The factoring algorithm finds B' as the most-shared literal, factors it out,
and places it once in series with a parallel sub-network of the remainders:

  Factored AB' + B'C  ->  series( B',  parallel(A, C) )            3 transistors

The PUN is then derived as the topological dual:

  PUN  ->  parallel( B',  series(A, C) )                           3 PMOS transistors

Further examples
~~~~~~~~~~~~~~~~
  AB + AC          ->  series( A,  parallel(B, C) )
  AB'C + B'CD      ->  series( B',  series( C, parallel(A, D) ) )
  A + B            ->  parallel( A, B )    (no sharing)

Module layout
-------------
  parse_sop(sop_str)              -> list[list[tuple[str, bool]]]
  NetworkNode                     -> recursive tree dataclass
  factor_pdn(terms)               -> NetworkNode  (NMOS, factored)
  build_pun(pdn_root)             -> NetworkNode  (PMOS, dual)
  CMOSCanvas(tk.Canvas)           -> full schematic renderer
  open_cmos_window(parent, expr)  -> convenience Toplevel launcher
"""

from __future__ import annotations

import tkinter as tk
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# Color palette  (matches the dark theme of the main app)
# ---------------------------------------------------------------------------
BG          = "#0f172a"
RAIL_COLOR  = "#f43f5e"
WIRE_COLOR  = "#94a3b8"
NMOS_COLOR  = "#38bdf8"
PMOS_COLOR  = "#a78bfa"
LABEL_COLOR = "#e5e7eb"
DIM_COLOR   = "#475569"
OUT_COLOR   = "#fbbf24"

FONT_LABEL = ("Segoe UI", 10, "bold")
FONT_TITLE = ("Segoe UI", 11, "bold")

# Type aliases
Literal = Tuple[str, bool]   # (variable_name, is_complemented)
Term    = List[Literal]


# ---------------------------------------------------------------------------
# 1.  SOP Parser
# ---------------------------------------------------------------------------

def parse_sop(sop_str: str) -> List[Term]:
    """
    Parse a minimized SOP string into structured product terms.

    "A B' + B' C"
    -> [ [("A",False), ("B",True)],
         [("B",True),  ("C",False)] ]

    "1" -> [[]]    tautology (one empty term)
    "0" -> []      constant zero (no terms)
    """
    s = (sop_str or "0").strip()

    if s == "0":
        return []
    if s == "1":
        return [[]]

    product_terms: List[Term] = []
    for raw in s.split("+"):
        raw = raw.strip()
        if not raw:
            continue
        literals: Term = []
        for tok in raw.split():
            tok = tok.strip()
            if not tok:
                continue
            if tok.endswith("'"):
                literals.append((tok[:-1], True))
            else:
                literals.append((tok, False))
        if literals:
            product_terms.append(literals)

    return product_terms


def _lit_label(lit: Literal) -> str:
    var, comp = lit
    return f"{var}'" if comp else var


def parse_pos(pos_str: str) -> List[Term]:
    """
    Parse a minimized POS string into structured sum terms.

    F' (POS) format produced by qm_algorithm.sop_to_pos():
        "(A' + B)(C')"  ->  [ [("A",True), ("B",False)],
                               [("C",True)] ]

    Each outer group (parenthesised factor) is one sum term (OR of literals).
    The product of all factors is the POS expression.

    "0" -> []    constant zero  (no factors)
    "1" -> [[]]  tautology      (one empty factor)
    """
    import re as _re
    s = (pos_str or "0").strip()
    if s == "0":
        return []
    if s == "1":
        return [[]]

    groups = _re.findall(r'\(([^)]+)\)', s)
    if not groups:
        groups = [s]          # no parens: single-literal POS e.g. "(A')"

    sum_terms: List[Term] = []
    for group in groups:
        literals: Term = []
        for tok in _re.split(r'\s*\+\s*', group.strip()):
            tok = tok.strip()
            if not tok:
                continue
            if tok.endswith("'"):
                literals.append((tok[:-1], True))
            else:
                literals.append((tok, False))
        if literals:
            sum_terms.append(literals)
    return sum_terms


def _make_leaf(lit: Literal, ttype: str) -> NetworkNode:
    return NetworkNode(kind="transistor", transistor_type=ttype, label=_lit_label(lit))


def _factor_sop_pun(terms: List[Term]) -> Optional[NetworkNode]:
    """
    Recursively factor shared literals from SOP product terms for PUN (PMOS).

    Rules:
        AND-term (product)  -> PMOS in SERIES
        OR of terms (sum)   -> series-groups in PARALLEL
        Shared literal L    -> one PMOS in SERIES with PARALLEL sub-network of remainders

    Example: F = A'B'C + A'BC' + AB'C' + ABC
        A' shared by (B'C, BC') and A shared by (B'C', BC)
        -> parallel(
             series(A', parallel(series(B',C), series(B,C'))),
             series(A,  parallel(series(B',C'), series(B,C)))
           )
        = 10 PMOS  (vs 12 naive)
    """
    if not terms:
        return None

    # Single term: build series chain directly
    if len(terms) == 1:
        term = terms[0]
        if not term:
            return NetworkNode(kind="transistor", transistor_type="pmos", label="1")
        if len(term) == 1:
            return _make_leaf(term[0], "pmos")
        return NetworkNode(kind="series", transistor_type="pmos",
                           children=[_make_leaf(l, "pmos") for l in term])

    # Find most-shared literal
    freq = _count_freq(terms)
    best_lit: Optional[Literal] = None
    best_cnt = 0
    for lit, cnt in freq.items():
        if cnt > best_cnt:
            best_cnt, best_lit = cnt, lit

    # No sharing: naive parallel of series
    if best_cnt <= 1:
        branches = []
        for term in terms:
            if not term:
                branches.append(NetworkNode(kind="transistor", transistor_type="pmos", label="1"))
            elif len(term) == 1:
                branches.append(_make_leaf(term[0], "pmos"))
            else:
                branches.append(NetworkNode(kind="series", transistor_type="pmos",
                                            children=[_make_leaf(l, "pmos") for l in term]))
        return (branches[0] if len(branches) == 1
                else NetworkNode(kind="parallel", transistor_type="pmos", children=branches))

    # Factor: shared literal goes in SERIES with PARALLEL sub-network of remainders
    matching     = [[l for l in t if l != best_lit] for t in terms if best_lit in t]
    non_matching = [t for t in terms if best_lit not in t]

    shared = _make_leaf(best_lit, "pmos")
    inner  = _factor_sop_pun(matching)   # parallel sub-network of what remains

    factored = shared if inner is None else NetworkNode(
        kind="series", transistor_type="pmos", children=[shared, inner])

    if not non_matching:
        return factored
    rest = _factor_sop_pun(non_matching)
    if rest is None:
        return factored
    return NetworkNode(kind="parallel", transistor_type="pmos", children=[factored, rest])


def factor_pun(product_terms: List[Term]) -> Optional[NetworkNode]:
    """Build the Pull-Up Network (PMOS) from F (SOP) with transistor sharing."""
    return _factor_sop_pun(product_terms)


def _factor_pos_pdn(terms: List[Term]) -> Optional[NetworkNode]:
    """
    Recursively factor shared literals from POS sum terms for PDN (NMOS).

    Rules:
        OR-term (sum)         -> NMOS in PARALLEL
        AND of terms (product)-> parallel-groups in SERIES
        Shared literal L      -> one NMOS in PARALLEL with SERIES sub-network of remainders

    Example: F' (POS) = (A+B+C')(A+B'+C)(A'+B+C)(A'+B'+C')
        A shared by first two terms, A' shared by last two
        -> series(
             parallel(A, series(parallel(B,C'), parallel(B',C))),
             parallel(A', series(parallel(B,C), parallel(B',C')))
           )
        = 10 NMOS  (vs 12 naive)
    """
    if not terms:
        return None

    # Single term: build parallel group directly
    if len(terms) == 1:
        term = terms[0]
        if not term:
            return NetworkNode(kind="transistor", transistor_type="nmos", label="1")
        if len(term) == 1:
            return _make_leaf(term[0], "nmos")
        return NetworkNode(kind="parallel", transistor_type="nmos",
                           children=[_make_leaf(l, "nmos") for l in term])

    # Find most-shared literal
    freq = _count_freq(terms)
    best_lit: Optional[Literal] = None
    best_cnt = 0
    for lit, cnt in freq.items():
        if cnt > best_cnt:
            best_cnt, best_lit = cnt, lit

    # No sharing: naive series of parallel groups
    if best_cnt <= 1:
        branches = []
        for term in terms:
            if not term:
                branches.append(NetworkNode(kind="transistor", transistor_type="nmos", label="1"))
            elif len(term) == 1:
                branches.append(_make_leaf(term[0], "nmos"))
            else:
                branches.append(NetworkNode(kind="parallel", transistor_type="nmos",
                                            children=[_make_leaf(l, "nmos") for l in term]))
        return (branches[0] if len(branches) == 1
                else NetworkNode(kind="series", transistor_type="nmos", children=branches))

    # Factor: shared literal goes in PARALLEL with SERIES sub-network of remainders
    matching     = [[l for l in t if l != best_lit] for t in terms if best_lit in t]
    non_matching = [t for t in terms if best_lit not in t]

    shared = _make_leaf(best_lit, "nmos")
    inner  = _factor_pos_pdn(matching)   # series sub-network of what remains

    factored = shared if inner is None else NetworkNode(
        kind="parallel", transistor_type="nmos", children=[shared, inner])

    if not non_matching:
        return factored
    rest = _factor_pos_pdn(non_matching)
    if rest is None:
        return factored
    return NetworkNode(kind="series", transistor_type="nmos", children=[factored, rest])


def factor_pdn_from_pos(sum_terms: List[Term]) -> Optional[NetworkNode]:
    """Build the Pull-Down Network (NMOS) from F' (POS) with transistor sharing."""
    return _factor_pos_pdn(sum_terms)



# ---------------------------------------------------------------------------
# 2.  Network tree
# ---------------------------------------------------------------------------

@dataclass
class NetworkNode:
    """
    Recursive tree node for a MOSFET network.

    kind values
    -----------
    "transistor"  leaf: one MOSFET
    "series"      children connected drain->source
    "parallel"    children share top/bottom rails

    transistor_type : "nmos" | "pmos"
    label           : gate signal string for transistor leaves
    children        : sub-nodes for series/parallel
    """
    kind: str
    transistor_type: str = "nmos"
    label: str = ""
    children: List["NetworkNode"] = field(default_factory=list)

    # Filled by the layout measurement pass; excluded from equality / repr
    _width:  int = field(default=0, repr=False, compare=False)
    _height: int = field(default=0, repr=False, compare=False)

def count_transistors(node: Optional[NetworkNode]) -> int:
    """Recursively count the number of transistor leaves in the tree."""
    if not node: return 0
    if node.kind == "transistor": return 1
    return sum(count_transistors(c) for c in node.children)


# ---------------------------------------------------------------------------
# 3.  Transistor-sharing PDN builder
# ---------------------------------------------------------------------------

def _count_freq(terms: List[Term]) -> Counter:
    """Count how many terms each literal appears in (at most once per term)."""
    freq: Counter = Counter()
    for term in terms:
        seen: set = set()
        for lit in term:
            if lit not in seen:
                freq[lit] += 1
                seen.add(lit)
    return freq


def _make_transistor(lit: Literal, ttype: str) -> NetworkNode:
    return NetworkNode(kind="transistor", transistor_type=ttype, label=_lit_label(lit))


def _make_series_chain(term: Term, ttype: str) -> NetworkNode:
    """Build a series chain from a single product term (no factoring needed)."""
    if not term:
        return NetworkNode(kind="transistor", transistor_type=ttype, label="1")
    if len(term) == 1:
        return _make_transistor(term[0], ttype)
    return NetworkNode(
        kind="series", transistor_type=ttype,
        children=[_make_transistor(l, ttype) for l in term],
    )


def _factor_terms(terms: List[Term], ttype: str) -> Optional[NetworkNode]:
    """
    Recursively factor the most-shared literal out of a list of product terms.

    Algorithm
    ---------
    1. Count literal frequency across all terms.
    2. Pick the literal L with the highest frequency.
    3. If L appears in only 1 term -> no sharing possible; build naive
       parallel-of-series and return.
    4. Otherwise split terms into:
         matching     = terms that contain L  (with L removed)
         non_matching = terms that do not contain L
    5. Recurse on matching remainders -> inner_node
    6. Build:  series( transistor(L),  inner_node )
    7. If non_matching is non-empty, wrap in:
         parallel( series(L, inner_node),  recurse(non_matching) )

    This is the "kernel extraction" technique from logic synthesis.

    Examples
    --------
    AB' + B'C   ->  series( B',  parallel(A, C) )
    AB  + AC    ->  series( A,   parallel(B, C) )
    AB'C + B'CD ->  series( B',  series(C,  parallel(A, D)) )
    A  + B      ->  parallel(A, B)                 (no sharing)
    """
    if not terms:
        return None

    # ---- single term: no factoring needed --------------------------------
    if len(terms) == 1:
        return _make_series_chain(terms[0], ttype)

    # ---- find most-shared literal ----------------------------------------
    freq     = _count_freq(terms)
    best_lit: Optional[Literal] = None
    best_cnt = 0
    for lit, cnt in freq.items():
        if cnt > best_cnt:
            best_cnt = cnt
            best_lit = lit

    # ---- no sharing: naive parallel-of-series ----------------------------
    if best_cnt <= 1:
        branches = [_make_series_chain(t, ttype) for t in terms]
        if len(branches) == 1:
            return branches[0]
        return NetworkNode(kind="parallel", transistor_type=ttype, children=branches)

    # ---- split and recurse -----------------------------------------------
    matching:     List[Term] = []
    non_matching: List[Term] = []

    for term in terms:
        if best_lit in term:
            matching.append([l for l in term if l != best_lit])
        else:
            non_matching.append(term)

    shared_t = _make_transistor(best_lit, ttype)
    inner    = _factor_terms(matching, ttype)

    # series( shared, inner )
    if inner is None:
        factored: NetworkNode = shared_t
    else:
        factored = NetworkNode(
            kind="series", transistor_type=ttype,
            children=[shared_t, inner],
        )

    if not non_matching:
        return factored

    # parallel( factored, recurse(non_matching) )
    rest = _factor_terms(non_matching, ttype)
    if rest is None:
        return factored
    return NetworkNode(kind="parallel", transistor_type=ttype,
                       children=[factored, rest])


def factor_pdn(terms: List[Term]) -> Optional[NetworkNode]:
    """Build the factored Pull-Down Network (NMOS) tree."""
    return _factor_terms(terms, "nmos")


# ---------------------------------------------------------------------------
# 4.  PUN builder  (topological dual of the factored PDN)
# ---------------------------------------------------------------------------

def _dual(node: NetworkNode) -> NetworkNode:
    """Derive the CMOS dual: series<->parallel, nmos->pmos, same labels."""
    if node.kind == "transistor":
        return NetworkNode(kind="transistor", transistor_type="pmos", label=node.label)
    elif node.kind == "series":
        return NetworkNode(kind="parallel", transistor_type="pmos",
                           children=[_dual(c) for c in node.children])
    elif node.kind == "parallel":
        return NetworkNode(kind="series", transistor_type="pmos",
                           children=[_dual(c) for c in node.children])
    return node


def build_pun(pdn_root: Optional[NetworkNode]) -> Optional[NetworkNode]:
    """Build the Pull-Up Network (PMOS) as the exact dual of the factored PDN."""
    return None if pdn_root is None else _dual(pdn_root)


# ---------------------------------------------------------------------------
# 5.  Canvas renderer
# ---------------------------------------------------------------------------

T_W    = 44    # transistor symbol width
T_H    = 52    # transistor symbol height
H_GAP  = 34    # horizontal gap between parallel branches
V_GAP  = 16    # vertical gap between series transistors
RAIL_H = 22    # power rail band height
MARGIN_Y = 60


class CMOSCanvas(tk.Canvas):
    """
    Tkinter Canvas that renders the full CMOS gate schematic.

    VDD
     |
    PUN (PMOS)
     |
    ─── F ───
     |
    PDN (NMOS)
     |
    GND
    """

    def __init__(self, parent: tk.Widget, **kwargs):
        super().__init__(parent, bg=BG, highlightthickness=0, **kwargs)
        self._pdn: Optional[NetworkNode] = None
        self._pun: Optional[NetworkNode] = None
        self._f_sop       = ""
        self._f_prime_sop = ""
        self.bind("<Configure>", self._on_resize)

    def load_expression(self, f_sop: str, f_prime_sop: str) -> None:
        """
        Build and render both networks using the correct CMOS convention:

          PDN (NMOS): built from F' (SOP)
              AND-terms (product terms) -> NMOS transistors in SERIES
              OR of terms (sum)         -> series-groups in PARALLEL

          PUN (PMOS): built as the exact topological dual of the PDN.
        """
        self._f_sop       = f_sop
        self._f_prime_sop = f_prime_sop

        # PDN: NMOS from F' (SOP)
        pdn_terms  = parse_sop(f_prime_sop)
        self._pdn  = factor_pdn(pdn_terms)

        # PUN: PMOS as exact dual of the PDN
        self._pun  = build_pun(self._pdn)

        self._render()

    def _on_resize(self, _e: tk.Event) -> None:
        self._render()

    # ------------------------------------------------------------------
    # Top-level render
    # ------------------------------------------------------------------

    def _render(self) -> None:
        self.delete("all")
        W = self.winfo_width()
        H = self.winfo_height()
        if W < 10 or H < 10:
            return

        if self._pdn is None and self._pun is None:
            self._draw_msg("F = 0  (output is never driven)", W, H)
            return

        pdn_w, pdn_h = self._measure(self._pdn) if self._pdn else (T_W, 0)
        pun_w, pun_h = self._measure(self._pun) if self._pun else (T_W, 0)
        net_w = max(pdn_w, pun_w, T_W)

        V_PADDING = 30
        total_h = RAIL_H + V_PADDING + pun_h + 42 + pdn_h + V_PADDING + RAIL_H + 18
        
        min_w_needed = net_w + 300  # padding for side labels
        eff_W = max(W, min_w_needed)
        eff_H = max(H, total_h + 100)
        
        cx    = eff_W // 2
        top_y = max(MARGIN_Y, (eff_H - total_h) // 2)

        rx1 = cx - net_w // 2 - 26
        rx2 = cx + net_w // 2 + 26

        # VDD
        vdd_y = top_y
        self._draw_rail(rx1, rx2, vdd_y, "VDD")

        # PUN  (built from F SOP, dualized → PMOS)
        pun_top = vdd_y + RAIL_H + V_PADDING
        if self._pun:
            self._draw_network(self._pun, cx, pun_top)
        pun_bot = pun_top + pun_h

        # Output node
        out_y = pun_bot + 21
        self._draw_output_node(cx, out_y, net_w)

        # PDN  (built from F' SOP → NMOS)
        pdn_top = out_y + 21
        if self._pdn:
            self._draw_network(self._pdn, cx, pdn_top)
        pdn_bot = pdn_top + pdn_h

        # GND
        gnd_y = pdn_bot + V_PADDING
        self._draw_rail(rx1, rx2, gnd_y, "GND")

        # Spine wires
        self.create_line(cx, vdd_y + RAIL_H - 4, cx, pun_top, fill=WIRE_COLOR, width=2)
        self.create_line(cx, pun_bot, cx, out_y, fill=WIRE_COLOR, width=2)
        self.create_line(cx, out_y, cx, pdn_top, fill=WIRE_COLOR, width=2)
        self.create_line(cx, pdn_bot, cx, gnd_y, fill=WIRE_COLOR, width=2)

        # Network labels on the right margin
        label_x = rx2 + 10
        if pun_h > 0:
            self.create_text(
                label_x, pun_top + pun_h // 2,
                text=f"PUN (PMOS)\nDual of PDN",
                fill=PMOS_COLOR, font=("Segoe UI", 8), anchor="w",
            )
        if pdn_h > 0:
            self.create_text(
                label_x, pdn_top + pdn_h // 2,
                text=f"PDN (NMOS)\nF' (SOP) = {self._f_prime_sop}",
                fill=NMOS_COLOR, font=("Segoe UI", 8), anchor="w",
            )

        # Title
        self.create_text(
            cx, top_y - 20,
            text=f"F (SOP) = {self._f_sop}     F' (SOP) = {self._f_prime_sop}",
            fill=OUT_COLOR, font=FONT_TITLE, anchor="s",
        )

        bbox = self.bbox("all")
        if bbox:
            self.configure(scrollregion=(bbox[0]-30, bbox[1]-30, bbox[2]+30, bbox[3]+30))

    # ------------------------------------------------------------------
    # Measurement pass
    # ------------------------------------------------------------------

    def _measure(self, node: NetworkNode) -> Tuple[int, int]:
        if node is None:
            return (T_W, T_H)

        if node.kind == "transistor":
            node._width, node._height = T_W, T_H
            return (T_W, T_H)

        if node.kind == "series":
            max_w = total_h = 0
            for i, child in enumerate(node.children):
                cw, ch = self._measure(child)
                max_w    = max(max_w, cw)
                total_h += ch + (V_GAP if i < len(node.children) - 1 else 0)
            node._width, node._height = max_w, total_h
            return (max_w, total_h)

        if node.kind == "parallel":
            total_w = max_h = 0
            for i, child in enumerate(node.children):
                cw, ch = self._measure(child)
                total_w += cw + (H_GAP if i < len(node.children) - 1 else 0)
                max_h    = max(max_h, ch)
            node._width, node._height = total_w, max_h
            return (total_w, max_h)

        return (T_W, T_H)

    # ------------------------------------------------------------------
    # Drawing pass  (returns bottom-Y of the drawn subtree)
    # ------------------------------------------------------------------

    def _draw_network(self, node: NetworkNode, cx: int, top_y: int) -> int:
        if node is None:
            return top_y
        if node.kind == "transistor":
            return self._draw_transistor(node, cx, top_y)
        if node.kind == "series":
            return self._draw_series(node, cx, top_y)
        if node.kind == "parallel":
            return self._draw_parallel(node, cx, top_y)
        return top_y + node._height

    # ---- MOSFET symbol ---------------------------------------------------

    def _draw_transistor(self, node: NetworkNode, cx: int, top_y: int) -> int:
        """
        Draw a simplified MOSFET symbol.

        NMOS                  PMOS
          D                     D
          |                     |
        G-|=                 G-o|=
          |                     |
          S                     S

        Channel bar at body_x (left of cx).
        Gate exits left with label.
        Drain/source vertical wires run through cx.
        """
        is_pmos  = (node.transistor_type == "pmos")
        color    = PMOS_COLOR if is_pmos else NMOS_COLOR

        body_x   = cx - 5
        chan_top  = top_y + 8
        chan_bot  = top_y + T_H - 8
        chan_mid  = (chan_top + chan_bot) // 2
        gate_x   = body_x - 22

        # Channel bar
        self.create_line(body_x, chan_top, body_x, chan_bot,
                         fill=color, width=5, capstyle=tk.ROUND)

        # Drain/source horizontal stubs
        self.create_line(cx, chan_top, body_x, chan_top, fill=color, width=2)
        self.create_line(cx, chan_bot, body_x, chan_bot, fill=color, width=2)

        # Drain/source vertical continuations
        self.create_line(cx, top_y,        cx, chan_top, fill=WIRE_COLOR, width=2)
        self.create_line(cx, chan_bot,      cx, top_y + T_H, fill=WIRE_COLOR, width=2)

        # Gate insulator bar
        ins_x = body_x - 5
        self.create_line(ins_x, chan_top, ins_x, chan_bot, fill=color, width=2)

        if is_pmos:
            br     = 5
            bub_cx = ins_x - 3 - br
            self.create_oval(bub_cx - br, chan_mid - br,
                             bub_cx + br, chan_mid + br,
                             outline=color, fill=BG, width=2)
            line_start = bub_cx - br
        else:
            line_start = ins_x - 3

        # Gate line
        self.create_line(line_start, chan_mid, gate_x, chan_mid, fill=color, width=2)

        # Gate label
        self.create_text(gate_x - 5, chan_mid, text=node.label,
                         fill=LABEL_COLOR, font=FONT_LABEL, anchor="e")

        return top_y + T_H

    # ---- series ----------------------------------------------------------

    def _draw_series(self, node: NetworkNode, cx: int, top_y: int) -> int:
        y = top_y
        for i, child in enumerate(node.children):
            y = self._draw_network(child, cx, y)
            if i < len(node.children) - 1:
                self.create_line(cx, y, cx, y + V_GAP, fill=WIRE_COLOR, width=2)
                y += V_GAP
        return y

    # ---- parallel --------------------------------------------------------

    def _draw_parallel(self, node: NetworkNode, cx: int, top_y: int) -> int:
        if not node.children:
            return top_y

        sizes   = [self._measure(c) for c in node.children]
        total_w = sum(w for w, _ in sizes) + H_GAP * (len(sizes) - 1)
        max_h   = max(h for _, h in sizes)

        x_cur = cx - total_w // 2
        tops: List[Tuple[int,int]] = []
        bots: List[Tuple[int,int]] = []

        for child, (cw, ch) in zip(node.children, sizes):
            ccx         = x_cur + cw // 2
            child_top   = top_y + (max_h - ch) // 2   # vertically centre
            child_bot   = child_top + ch
            self._draw_network(child, ccx, child_top)
            tops.append((ccx, child_top))
            bots.append((ccx, child_bot))
            x_cur += cw + H_GAP

        top_bus = top_y
        bot_bus = top_y + max_h

        if len(node.children) > 1:
            lx = tops[0][0]
            rx = tops[-1][0]
            self.create_line(lx, top_bus, rx, top_bus, fill=WIRE_COLOR, width=2)
            self.create_line(lx, bot_bus, rx, bot_bus, fill=WIRE_COLOR, width=2)
            for (ccx, ct), (_, cb) in zip(tops, bots):
                self.create_line(ccx, top_bus, ccx, ct, fill=WIRE_COLOR, width=2)
                self.create_line(ccx, cb,      ccx, bot_bus, fill=WIRE_COLOR, width=2)

        return bot_bus

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _draw_rail(self, x1: int, x2: int, y: int, label: str) -> None:
        bh = RAIL_H - 4
        self.create_rectangle(x1, y, x2, y + bh,
                              fill="#1e2d47", outline=RAIL_COLOR, width=2)
        self.create_text((x1 + x2) // 2, y + bh // 2,
                         text=label, fill=RAIL_COLOR, font=FONT_TITLE)

    def _draw_output_node(self, cx: int, y: int, net_w: int) -> None:
        half = net_w // 2 + 26
        self.create_line(cx - half, y, cx + half, y,
                         fill=OUT_COLOR, width=2, dash=(6, 3))
        r = 5
        self.create_oval(cx - r, y - r, cx + r, y + r,
                         fill=OUT_COLOR, outline=OUT_COLOR)
        self.create_text(cx + half + 10, y, text="F",
                         fill=OUT_COLOR, font=FONT_TITLE, anchor="w")

    def _draw_msg(self, msg: str, W: int, H: int) -> None:
        self.create_text(W // 2, H // 2, text=msg,
                         fill=LABEL_COLOR, font=FONT_TITLE)


# ---------------------------------------------------------------------------
# 6.  Convenience window launcher
# ---------------------------------------------------------------------------

def open_cmos_window(parent: tk.Widget, f_sop: str, f_prime_sop: str,
                     title: str = "CMOS Circuit") -> None:
    """
    Open a Toplevel window with the full CMOS schematic.

    Parameters
    ----------
    parent       : parent Tk widget
    f_sop        : minimized SOP for F   — used to build PUN (PMOS)
    f_prime_sop  : minimized SOP for F'  — used to build PDN (NMOS)
    title        : window title
    """
    win = tk.Toplevel(parent)
    win.title(title)
    win.geometry("960x700")
    win.minsize(720, 540)
    win.configure(bg=BG)

    # Header
    header = tk.Frame(win, bg="#111827", pady=8)
    header.pack(fill="x")
    tk.Label(header,
             text="CMOS Transistor Implementation  –  PUN / PDN Dual Network",
             bg="#111827", fg="#e5e7eb",
             font=("Segoe UI", 12, "bold")).pack(side="left", padx=16)
    # Show both expressions
    expr_frame = tk.Frame(header, bg="#111827")
    expr_frame.pack(side="right", padx=16)
    tk.Label(expr_frame, text=f"F  (SOP)  =  {f_sop}",
             bg="#111827", fg=PMOS_COLOR,
             font=("Segoe UI", 11)).pack(anchor="e")
    tk.Label(expr_frame, text=f"F' (SOP)  =  {f_prime_sop}",
             bg="#111827", fg=NMOS_COLOR,
             font=("Segoe UI", 11)).pack(anchor="e")

    # Legend
    legend = tk.Frame(win, bg="#0b1220", pady=6)
    legend.pack(fill="x")

    def _legend_item(frame, color, text):
        f = tk.Frame(frame, bg="#0b1220")
        f.pack(side="left", padx=12)
        c = tk.Canvas(f, width=20, height=14, bg="#0b1220", highlightthickness=0)
        c.pack(side="left")
        c.create_rectangle(2, 3, 18, 11, fill=color, outline=color)
        tk.Label(f, text=text, bg="#0b1220", fg="#cbd5e1",
                 font=("Segoe UI", 9)).pack(side="left")

    _legend_item(legend, PMOS_COLOR, "PMOS  (PUN – Exact topological dual of PDN)")
    _legend_item(legend, NMOS_COLOR, "NMOS  (PDN – F' in SOP:  AND-terms → series NMOS,  OR of terms → parallel)")
    _legend_item(legend, RAIL_COLOR, "VDD / GND power rails")
    _legend_item(legend, OUT_COLOR,  "Output node F")

    # Note
    note = tk.Frame(win, bg="#0b1220", pady=4)
    note.pack(fill="x")
    tk.Label(
        note,
        text=(
            "PUN (PMOS): Exact dual of PDN (parallel converted to series, and vice versa)  |  "
            "PDN (NMOS): F' (SOP) — AND-terms → series NMOS,  OR of terms → parallel  |  "
            "Shared literals factored out"
        ),
        bg="#0b1220", fg=DIM_COLOR, font=("Segoe UI", 9),
    ).pack()

    # Canvas
    canvas_frame = tk.Frame(win, bg=BG)
    canvas_frame.pack(fill="both", expand=True, padx=12, pady=12)
    
    v_scroll = tk.Scrollbar(canvas_frame, orient="vertical")
    v_scroll.pack(side="right", fill="y")
    
    h_scroll = tk.Scrollbar(canvas_frame, orient="horizontal")
    h_scroll.pack(side="bottom", fill="x")

    cmos = CMOSCanvas(canvas_frame, yscrollcommand=v_scroll.set, xscrollcommand=h_scroll.set)
    cmos.pack(side="left", fill="both", expand=True)

    v_scroll.config(command=cmos.yview)
    h_scroll.config(command=cmos.xview)

    win.update_idletasks()
    cmos.load_expression(f_sop, f_prime_sop)
