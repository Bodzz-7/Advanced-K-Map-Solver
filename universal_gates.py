import tkinter as tk
from tkinter import ttk
from cmos_schematic import parse_sop, parse_pos, _lit_label

BG          = "#1e1e1e"
GATE_BG     = "#2d2d2d"
GATE_FG     = "#ffffff"
WIRE_COLOR  = "#64748b"
LIT_COLOR   = "#38bdf8"
TITLE_COLOR = "#a78bfa"

class UniversalGatesWindow(tk.Toplevel):
    def __init__(self, parent, f_sop: str, f_pos: str):
        super().__init__(parent)
        self.title("Universal Gates Synthesizer")
        self.geometry("960x700")
        self.minsize(800, 600)
        self.configure(bg="#1c1c1c")

        self.f_sop = f_sop.strip() or "0"
        self.f_pos = f_pos.strip() or "0"

        

        style = ttk.Style(self)
        style.configure("UG.TNotebook", background=BG, borderwidth=0)
        style.configure("UG.TNotebook.Tab", padding=(12, 8), font=("Segoe UI", 11, "bold"))
        
        self.notebook = ttk.Notebook(self, style="UG.TNotebook")
        self.notebook.pack(fill="both", expand=True, padx=12, pady=12)

        self.tab_nand = ttk.Frame(self.notebook)
        self.tab_nor = ttk.Frame(self.notebook)

        self.notebook.add(self.tab_nand, text="  NAND Synthesis (from SOP)  ")
        self.notebook.add(self.tab_nor, text="  NOR Synthesis (from POS)  ")

        self._build_nand_tab()
        self._build_nor_tab()

    def _build_nand_tab(self):
        header = ttk.Label(self.tab_nand, text=f"F (SOP) = {self.f_sop}", foreground=TITLE_COLOR, font=("Segoe UI Variable Display", 14, "bold"), padding=10)
        header.pack()

        self.canvas_nand = tk.Canvas(self.tab_nand, bg=BG, highlightthickness=0)
        self.canvas_nand.pack(fill="both", expand=True)
        self.canvas_nand.bind("<Configure>", lambda e: self._draw_nand())

    def _draw_nand(self):
        self.canvas_nand.delete("all")
        if self.f_sop in ("0", "1"):
            self._draw_msg(self.canvas_nand, f"F = {self.f_sop} (Trivial, no gates needed)")
            return
        terms = parse_sop(self.f_sop)
        self._draw_2level_graph(self.canvas_nand, terms, "NAND")

    def _build_nor_tab(self):
        header = ttk.Label(self.tab_nor, text=f"F (POS) = {self.f_pos}", foreground=TITLE_COLOR, font=("Segoe UI Variable Display", 14, "bold"), padding=10)
        header.pack()

        self.canvas_nor = tk.Canvas(self.tab_nor, bg=BG, highlightthickness=0)
        self.canvas_nor.pack(fill="both", expand=True)
        self.canvas_nor.bind("<Configure>", lambda e: self._draw_nor())

    def _draw_nor(self):
        self.canvas_nor.delete("all")
        if self.f_pos in ("0", "1"):
            self._draw_msg(self.canvas_nor, f"F = {self.f_pos} (Trivial, no gates needed)")
            return
        terms = parse_pos(self.f_pos)
        self._draw_2level_graph(self.canvas_nor, terms, "NOR")

    def _draw_msg(self, canvas: tk.Canvas, msg: str):
        canvas.update_idletasks()
        w = canvas.winfo_width() or 800
        h = canvas.winfo_height() or 500
        canvas.create_text(w//2, h//2, text=msg, fill=GATE_FG, font=("Segoe UI", 12))

    def _draw_2level_graph(self, canvas: tk.Canvas, terms: list, gate_type: str):
        """
        Draws a generic 2-level gate graph.
        terms: list of lists of literals.
        gate_type: "NAND" or "NOR".
        """
        W = canvas.winfo_width()
        H = canvas.winfo_height()
        if W < 10 or H < 10:
            return

        # Geometry
        # Level 1 nodes x = W * 0.4
        # Level 2 node x = W * 0.75
        # Inputs x = W * 0.1

        x_in = int(W * 0.1)
        x_l1 = int(W * 0.4)
        x_l2 = int(W * 0.75)

        y_center = H // 2

        num_terms = len(terms)
        if num_terms == 0:
            return
            
        spacing = max(60, min(100, H // (num_terms + 1)))
        
        l1_nodes = []
        start_y = y_center - (num_terms - 1) * spacing // 2

        for i, term in enumerate(terms):
            y = start_y + i * spacing
            inputs = [_lit_label(lit) for lit in term]
            l1_nodes.append((x_l1, y, inputs))

        # Draw Level 2 Gate
        l2_y = y_center
        # If there's only 1 term, Level 2 acts as an inverter (tied inputs).
        self._draw_gate(canvas, x_l2, l2_y, gate_type)
        
        # Draw final output line
        self._draw_wire(canvas, x_l2 + 30, l2_y, x_l2 + 80, l2_y)
        canvas.create_text(x_l2 + 90, l2_y, text="F", fill=TITLE_COLOR, font=("Segoe UI", 12, "bold"), anchor="w")

        # Connect Level 1 to Level 2
        for (nx, ny, inps) in l1_nodes:
            self._draw_gate(canvas, nx, ny, gate_type)
            self._draw_wire(canvas, nx + 30, ny, x_l2 - 30, l2_y)
            
            # Connect inputs to Level 1
            if len(inps) == 1:
                # Tied inputs to act as inverter
                self._draw_wire(canvas, x_in, ny - 10, nx - 30, ny - 10, inps[0])
                self._draw_wire(canvas, x_in, ny + 10, nx - 30, ny + 10, inps[0])
            else:
                inp_spacing = 30 / max(1, len(inps)-1) if len(inps) > 1 else 0
                start_in_y = ny - 15
                for j, inp in enumerate(inps):
                    in_y = start_in_y + j * inp_spacing
                    self._draw_wire(canvas, x_in, in_y, nx - 30, in_y, inp)

    def _draw_gate(self, canvas: tk.Canvas, x: int, y: int, label: str):
        # A sleek modern rounded rect for the gate
        w, h = 60, 40
        x1, y1 = x - w//2, y - h//2
        x2, y2 = x + w//2, y + h//2
        
        r = 10 # corner radius
        # draw a rounded rect using polygon
        pts = [
            x1+r, y1,  x2-r, y1,
            x2, y1,    x2, y1+r,
            x2, y2-r,  x2, y2,
            x2-r, y2,  x1+r, y2,
            x1, y2,    x1, y2-r,
            x1, y1+r,  x1, y1
        ]
        canvas.create_polygon(pts, fill=GATE_BG, outline=WIRE_COLOR, smooth=True, width=2)
        
        # small inversion bubble at the output
        bx, by = x2 + 5, y
        br = 4
        canvas.create_oval(bx-br, by-br, bx+br, by+br, fill=BG, outline=WIRE_COLOR, width=2)
        
        canvas.create_text(x, y, text=label, fill=GATE_FG, font=("Segoe UI", 10, "bold"))

    def _draw_wire(self, canvas: tk.Canvas, x1: int, y1: int, x2: int, y2: int, label: str = None):
        # Draw a sleek bezier curve wire
        # Control points to make it horizontal at the ends
        cx1 = x1 + (x2 - x1) // 2
        cy1 = y1
        cx2 = x1 + (x2 - x1) // 2
        cy2 = y2
        canvas.create_line(x1, y1, cx1, cy1, cx2, cy2, x2, y2, smooth=True, fill=WIRE_COLOR, width=2)

        if label:
            canvas.create_text(x1 - 10, y1, text=label, fill=LIT_COLOR, font=("Consolas", 11, "bold"), anchor="e")

def open_universal_gates_window(parent: tk.Widget, f_sop: str, f_pos: str):
    win = UniversalGatesWindow(parent, f_sop, f_pos)
    win.grab_set()
