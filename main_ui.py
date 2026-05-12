import tkinter as tk
from tkinter import ttk

from kmap_visuals import KMapCanvas
from qm_algorithm import QuineMcCluskey, get_all_derivations, get_all_derivations_from_solution
from cmos_schematic import open_cmos_window
from universal_gates import open_universal_gates_window


class KMapMinimizerApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("K-Map Logic Minimizer")
        self.minsize(1100, 640)
        
        try:
            self.iconbitmap(default="icon.ico")
        except tk.TclError:
            pass

        self._setup_style()

        self.var_count = tk.IntVar(value=4)
        self.results_vars = {
            "f_sop": tk.StringVar(value="—"),
            "f_sop_factored": tk.StringVar(value="—"),
            "f_pos": tk.StringVar(value="—"),
            "f_prime_sop": tk.StringVar(value="—"),
            "f_prime_sop_factored": tk.StringVar(value="—"),
            "f_prime_pos": tk.StringVar(value="—"),
        }
        self.show_pos_var = tk.BooleanVar(value=False)

        # Centralized truth-table/K-map state:
        # each index holds '0', '1', or 'X' (don't care).
        self.cell_states: list[str] = []

        # Mapping between K-map (row, col) and truth table indices (minterm indices)
        self.kmap_rc_to_index: dict[tuple[int, int], int] = {}
        self.index_to_kmap_rc: dict[int, tuple[int, int]] = {}

        # Truth table widgets
        self.f_buttons: list[ttk.Button] = []

        # Latest solver outputs (used for Algebra Steps window)
        self.last_expressions: dict[str, object] | None = None

        root = ttk.Frame(self, padding=14)
        root.grid(row=0, column=0, sticky="nsew")
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(1, weight=1)
        root.rowconfigure(2, weight=0)

        self._build_top_controls(root)
        self._build_main_area(root)
        self._build_results_frame(root)

        self._on_var_count_changed(initial=True)

    # -----------------------
    # UI construction
    # -----------------------

    def _setup_style(self) -> None:
        import sv_ttk
        sv_ttk.set_theme("dark")

        style = ttk.Style(self)
        
        # Configure fonts and specific tweaks
        style.configure(".", font=("Segoe UI Variable Display", 11))
        style.configure("Top.TLabel", font=("Segoe UI Variable Display", 12))
        
        style.configure(
            "Accent.TButton",
            font=("Segoe UI Variable Display", 11, "bold"),
        )
        
        style.configure(
            "FCell.TButton",
            font=("Segoe UI Variable Display", 11, "bold"),
            padding=(6, 4),
        )

        style.configure(
            "Header.TLabel",
            font=("Segoe UI Variable Display", 11, "bold"),
            padding=(10, 8),
            anchor="center",
        )
        style.configure(
            "Cell.TLabel",
            font=("Consolas", 11),
            padding=(10, 8),
            anchor="center",
        )
        style.configure("ResultsKey.TLabel", font=("Segoe UI Variable Display", 11))
        style.configure("ResultsVal.TLabel", font=("Segoe UI Variable Display", 12, "bold"))

    def _build_top_controls(self, parent: ttk.Frame) -> None:
        top = ttk.Frame(parent)
        top.grid(row=0, column=0, sticky="ew", padx=2, pady=(0, 12))
        top.columnconfigure(3, weight=1)

        ttk.Label(top, text="K-Map Logic Minimizer", style="Top.TLabel", font=("Segoe UI", 14, "bold")).grid(
            row=0, column=0, sticky="w", padx=(0, 18)
        )
        ttk.Label(top, text="Variables", style="Top.TLabel").grid(row=0, column=1, sticky="w", padx=(0, 8))

        self.var_combo = ttk.Combobox(
            top,
            values=(2, 3, 4),
            width=6,
            state="readonly",
            textvariable=self.var_count,
            font=("Segoe UI", 11),
        )
        self.var_combo.grid(row=0, column=2, sticky="w", padx=(0, 14))
        self.var_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_var_count_changed())

        ttk.Button(top, text="Solve", style="Accent.TButton", command=self.solve).grid(row=0, column=4, sticky="e")

    def _build_main_area(self, parent: ttk.Frame) -> None:
        main = ttk.Frame(parent)
        main.grid(row=1, column=0, sticky="nsew")
        main.columnconfigure(0, weight=1)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(0, weight=1)

        # Left: Truth Table (scrollable)
        left_border = ttk.Frame(main)
        left_border.grid(row=0, column=0, sticky="nsew", padx=(6, 10), pady=6)
        left_border.columnconfigure(0, weight=1)
        left_border.rowconfigure(0, weight=1)

        left = ttk.Labelframe(left_border, text="Truth Table", padding=10)
        left.grid(row=0, column=0, sticky="nsew", padx=1, pady=1)
        left.columnconfigure(0, weight=1)
        left.rowconfigure(0, weight=1)

        self.tt_canvas = tk.Canvas(left, bg="#1e1e1e", highlightthickness=0)
        self.tt_canvas.grid(row=0, column=0, sticky="nsew")
        tt_vbar = ttk.Scrollbar(left, orient="vertical", command=self.tt_canvas.yview)
        tt_vbar.grid(row=0, column=1, sticky="ns")
        self.tt_canvas.configure(yscrollcommand=tt_vbar.set)

        self.table_frame = tk.Frame(self.tt_canvas, bg="#505050")
        self._table_window = self.tt_canvas.create_window((0, 0), window=self.table_frame, anchor="nw")
        self.table_frame.bind("<Configure>", self._on_tt_configure)
        self.tt_canvas.bind("<Configure>", self._on_tt_canvas_configure)

        # Right: K-Map
        right_border = ttk.Frame(main)
        right_border.grid(row=0, column=1, sticky="nsew", padx=(10, 6), pady=6)
        right_border.columnconfigure(0, weight=1)
        right_border.rowconfigure(0, weight=1)

        right = ttk.Labelframe(right_border, text="K-Map", padding=10)
        right.grid(row=0, column=0, sticky="nsew", padx=1, pady=1)
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)

        self.kmap = KMapCanvas(right, self.var_count.get())
        self.kmap.grid(row=0, column=0, sticky="nsew")
        self.kmap.set_cell_click_callback(self._on_kmap_cell_clicked)

    def _build_results_frame(self, parent: ttk.Frame) -> None:
        results = ttk.Labelframe(parent, text="Results", padding=10)
        results.grid(row=2, column=0, sticky="ew", pady=(12, 0))
        results.columnconfigure(1, weight=1)

        self.pos_row_widgets = []

        rows = [
            ("F (SOP):", "f_sop", False),
            ("F (Factored):", "f_sop_factored", False),
            ("F (POS):", "f_pos", True),
            ("F' (SOP):", "f_prime_sop", False),
            ("F' (Factored):", "f_prime_sop_factored", False),
            ("F' (POS):", "f_prime_pos", True),
        ]

        for r, (label, key, is_pos) in enumerate(rows):
            lbl_key = ttk.Label(results, text=label, style="ResultsKey.TLabel")
            lbl_key.grid(row=r, column=0, sticky="w", padx=(2, 10))
            lbl_val = ttk.Label(results, textvariable=self.results_vars[key], style="ResultsVal.TLabel")
            lbl_val.grid(row=r, column=1, sticky="ew")
            
            if is_pos:
                self.pos_row_widgets.append((lbl_key, lbl_val))
                lbl_key.grid_remove()
                lbl_val.grid_remove()

        actions = ttk.Frame(results)
        actions.grid(row=len(rows), column=0, columnspan=2, sticky="ew", pady=(10, 0), padx=2)
        actions.columnconfigure(0, weight=1)

        pos_check = ttk.Checkbutton(
            actions,
            text="Show POS forms",
            variable=self.show_pos_var,
            command=self._toggle_pos_visibility
        )
        pos_check.grid(row=0, column=0, sticky="w")

        btn_frame = ttk.Frame(actions)
        btn_frame.grid(row=0, column=1, sticky="e")

        self.algebra_button = ttk.Button(
            btn_frame,
            text="View Algebra Steps",
            command=self.show_algebra_steps,
            state="disabled",
        )
        self.algebra_button.grid(row=0, column=0, sticky="e", padx=(0, 8))

        self.cmos_button = ttk.Button(
            btn_frame,
            text="⚡ Show CMOS Circuit",
            command=self.show_cmos_circuit,
            state="disabled",
        )
        self.cmos_button.grid(row=0, column=1, sticky="e")

        self.ug_button = ttk.Button(
            btn_frame,
            text="Synthesize using Universal Gates",
            command=self.show_universal_gates,
            state="disabled",
        )
        self.ug_button.grid(row=0, column=2, sticky="e", padx=(8, 0))

    def _toggle_pos_visibility(self):
        if self.show_pos_var.get():
            for key_lbl, val_lbl in self.pos_row_widgets:
                key_lbl.grid()
                val_lbl.grid()
        else:
            for key_lbl, val_lbl in self.pos_row_widgets:
                key_lbl.grid_remove()
                val_lbl.grid_remove()

    # -----------------------
    # Truth table behavior
    # -----------------------

    def _on_tt_configure(self, _event: tk.Event) -> None:
        self.tt_canvas.configure(scrollregion=self.tt_canvas.bbox("all"))

    def _on_tt_canvas_configure(self, event: tk.Event) -> None:
        self.tt_canvas.itemconfigure(self._table_window, width=event.width)

    def _on_var_count_changed(self, initial: bool = False) -> None:
        n = int(self.var_count.get())
        self._configure_kmap_index_mapping(n)

        # Reset state when variable count changes (or on first load).
        self.cell_states = ["0"] * (2**n)

        self._regenerate_truth_table()
        self.kmap.draw_grid(n)
        self.kmap.set_cell_click_callback(self._on_kmap_cell_clicked)
        self._sync_all_to_kmap()
        self._clear_results()
        if not initial:
            self.tt_canvas.yview_moveto(0.0)

    def _clear_results(self) -> None:
        for v in self.results_vars.values():
            v.set("—")
        self.last_expressions = None
        if hasattr(self, "algebra_button"):
            self.algebra_button.configure(state="disabled")
        if hasattr(self, "cmos_button"):
            self.cmos_button.configure(state="disabled")
        if hasattr(self, "ug_button"):
            self.ug_button.configure(state="disabled")

    def _configure_kmap_index_mapping(self, n: int) -> None:
        """
        Use the exact mappings provided by the user:

        2 Vars:
            {(0,0):0,(0,1):1,(1,0):2,(1,1):3}
        3 Vars:
            {(0,0):0,(0,1):1,(0,2):3,(0,3):2,(1,0):4,(1,1):5,(1,2):7,(1,3):6}
        4 Vars:
            {(0,0):0,(0,1):1,(0,2):3,(0,3):2,(1,0):4,(1,1):5,(1,2):7,(1,3):6,
             (2,0):12,(2,1):13,(2,2):15,(2,3):14,(3,0):8,(3,1):9,(3,2):11,(3,3):10}
        """

        if n == 2:
            self.kmap_rc_to_index = {(0, 0): 0, (0, 1): 1, (1, 0): 2, (1, 1): 3}
        elif n == 3:
            self.kmap_rc_to_index = {
                (0, 0): 0,
                (0, 1): 1,
                (0, 2): 3,
                (0, 3): 2,
                (1, 0): 4,
                (1, 1): 5,
                (1, 2): 7,
                (1, 3): 6,
            }
        elif n == 4:
            self.kmap_rc_to_index = {
                (0, 0): 0,
                (0, 1): 1,
                (0, 2): 3,
                (0, 3): 2,
                (1, 0): 4,
                (1, 1): 5,
                (1, 2): 7,
                (1, 3): 6,
                (2, 0): 12,
                (2, 1): 13,
                (2, 2): 15,
                (2, 3): 14,
                (3, 0): 8,
                (3, 1): 9,
                (3, 2): 11,
                (3, 3): 10,
            }
        else:
            self.kmap_rc_to_index = {}

        self.index_to_kmap_rc = {idx: rc for rc, idx in self.kmap_rc_to_index.items()}

    def _clear_truth_table(self) -> None:
        for child in self.table_frame.winfo_children():
            child.destroy()
        self.f_buttons.clear()

    def _regenerate_truth_table(self) -> None:
        self._clear_truth_table()

        n = int(self.var_count.get())
        var_names = ["A", "B", "C", "D"][:n]
        headers = [*var_names, "F"]

        for c, name in enumerate(headers):
            tk.Label(self.table_frame, text=name, font=("Segoe UI Variable Display", 11, "bold"), bg="#252525", fg="#ffffff", padx=10, pady=8).grid(
                row=0, column=c, sticky="nsew", padx=1, pady=1
            )
            self.table_frame.columnconfigure(c, weight=1)

        row_count = 2**n

        for r in range(row_count):
            bits = self._row_to_bits(r, n)
            for c in range(n):
                tk.Label(self.table_frame, text=str(bits[c]), font=("Consolas", 11), bg="#1e1e1e", fg="#ffffff", padx=10, pady=8).grid(
                    row=r + 1, column=c, sticky="nsew", padx=1, pady=1
                )

            btn = ttk.Button(
                self.table_frame,
                text=self.cell_states[r] if r < len(self.cell_states) else "0",
                style="FCell.TButton",
                command=lambda idx=r: self._cycle_state(idx),
                width=4,
            )
            btn.grid(row=r + 1, column=n, sticky="nsew", padx=1, pady=1)
            self.f_buttons.append(btn)

        self._sync_all_to_kmap()

    @staticmethod
    def _row_to_bits(row_index: int, n: int) -> list[int]:
        return [int(x) for x in format(row_index, f"0{n}b")]

    def _cycle_state(self, index: int) -> None:
        current = self.cell_states[index]
        nxt = {"0": "1", "1": "X", "X": "0"}[current]
        self.update_state(index, nxt)

    def update_state(self, index: int, new_value: str) -> None:
        """
        Single sync point for BOTH Truth Table and K-map clicks.

        Updates:
        - self.cell_states[index]
        - Truth Table button text at that index
        - K-map button text at corresponding (row, col)
        """

        if new_value not in ("0", "1", "X"):
            return
        if index < 0 or index >= len(self.cell_states):
            return

        self.cell_states[index] = new_value

        # Update truth table button
        if 0 <= index < len(self.f_buttons):
            self.f_buttons[index].configure(text=new_value)

        # Update kmap cell button
        rc = self.index_to_kmap_rc.get(index)
        if rc is not None:
            self.kmap.set_cell_value(rc[0], rc[1], new_value)

    def _sync_all_to_kmap(self) -> None:
        for idx, val in enumerate(self.cell_states):
            rc = self.index_to_kmap_rc.get(idx)
            if rc is not None:
                self.kmap.set_cell_value(rc[0], rc[1], val)

    def _on_kmap_cell_clicked(self, row: int, col: int) -> None:
        idx = self.kmap_rc_to_index.get((row, col))
        if idx is None:
            return
        self._cycle_state(idx)

    # -----------------------
    # Solve wiring
    # -----------------------

    def solve(self) -> None:
        n = int(self.var_count.get())

        minterms: list[int] = []
        maxterms: list[int] = []
        dont_cares: list[int] = []

        for i, v in enumerate(self.cell_states):
            if v == "1":
                minterms.append(i)
            elif v == "0":
                maxterms.append(i)
            elif v == "X":
                dont_cares.append(i)

        qm = QuineMcCluskey(n)
        expressions = qm.get_all_expressions(minterms=minterms, maxterms=maxterms, dont_cares=dont_cares)
        self.last_expressions = expressions

        self.results_vars["f_sop"].set(expressions["f_sop"])
        self.results_vars["f_sop_factored"].set(expressions["f_sop_factored"])
        self.results_vars["f_pos"].set(expressions["f_pos"])
        self.results_vars["f_prime_sop"].set(expressions["f_prime_sop"])
        self.results_vars["f_prime_sop_factored"].set(expressions["f_prime_sop_factored"])
        self.results_vars["f_prime_pos"].set(expressions["f_prime_pos"])

        if hasattr(self, "algebra_button"):
            self.algebra_button.configure(state="normal")

        if hasattr(self, "cmos_button"):
            self.cmos_button.configure(state="normal")
            
        if hasattr(self, "ug_button"):
            self.ug_button.configure(state="normal")

        # K-map values are already bound via update_state; still ensure sync.
        self._sync_all_to_kmap()

    def show_algebra_steps(self) -> None:
        if self.last_expressions is not None:
            derivation_text = get_all_derivations_from_solution(self.last_expressions)
        else:
            # Fallback (shouldn't happen if button is enabled only after solve)
            f_sop = self.results_vars["f_sop"].get()
            f_prime_sop = self.results_vars["f_prime_sop"].get()
            derivation_text = get_all_derivations(f_sop, f_prime_sop)

        win = tk.Toplevel(self)
        win.title("Boolean Algebra Derivations")
        win.geometry("800x600")
        win.minsize(800, 600)

        container = ttk.Frame(win, padding=10)
        container.grid(row=0, column=0, sticky="nsew")
        win.columnconfigure(0, weight=1)
        win.rowconfigure(0, weight=1)
        container.columnconfigure(0, weight=1)
        container.rowconfigure(0, weight=1)

        scrollbar = ttk.Scrollbar(container, orient="vertical")
        scrollbar.grid(row=0, column=1, sticky="ns")

        text = tk.Text(
            container,
            wrap=tk.WORD,
            font=("Consolas", 11),
            yscrollcommand=scrollbar.set,
            padx=12,
            pady=12,
        )
        text.grid(row=0, column=0, sticky="nsew")
        scrollbar.configure(command=text.yview)

        text.tag_configure("body", spacing1=2, spacing2=2, spacing3=6)
        text.insert("1.0", derivation_text if derivation_text else "(no derivation available)", ("body",))
        text.configure(state="disabled")


    def show_cmos_circuit(self) -> None:
        """
        Open the CMOS schematic window.

        PDN (NMOS) is built from F’ (SOP) — conducts when F = 0.
        PUN (PMOS) is built from the dual of F (SOP) — conducts when F = 1.
        """
        if self.last_expressions is None:
            return

        f_sop       = str(self.last_expressions.get("f_sop",       "0")).strip()
        f_prime_sop = str(self.last_expressions.get("f_prime_sop", "0")).strip()

        n     = self.last_expressions.get("num_variables", "?")
        title = f"CMOS Circuit  –  {n}-variable  |  F = {f_sop}"

        open_cmos_window(self, f_sop, f_prime_sop, title=title)

    def show_universal_gates(self) -> None:
        """Open the Universal Gates Synthesizer window."""
        if self.last_expressions is None:
            return

        f_sop = str(self.last_expressions.get("f_sop", "0")).strip()
        f_pos = str(self.last_expressions.get("f_pos", "0")).strip()

        open_universal_gates_window(self, f_sop, f_pos)


def main() -> None:
    app = KMapMinimizerApp()
    app.mainloop()


if __name__ == "__main__":
    main()

