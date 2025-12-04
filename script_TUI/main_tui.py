# Import existing logic
from textual.app import App, ComposeResult
from textual.containers import Horizontal, VerticalScroll, Container
from textual.widgets import Header, Footer, Button, Static, RichLog, Input, Label, ListItem, ListView
from textual.screen import Screen
from textual import on
import os
import sys

# Import existing logic
# We need to make sure we can import from the current directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from file_reader import read_graph_from_file as load_graph_from_file
from test_sec import (
    dfs, bfs_traversal, preorder, postorder, inorder, level_order,
    get_shortest_path, get_longest_path, longest_path_with_cycles,
    minimum_cut, number_of_spanning_trees, maximum_spanning_tree,
    minimum_spanning_tree, get_safest_path, get_widest_path,
    get_narrowest_path, get_most_dangerous_path,
    get_safest_path_by_boreccz1, get_most_dangerous_path_by_boreccz1,
    moore_shortest_path
)
from matrix_operations import (
    adjacency_matrix,
    incidence_matrix,
    matrix_power,
    laplacian_matrix,
    distance_matrix,
    predecessor_matrix,
    sign_matrix,
    save_matrix_to_file
)
from graph import (
    Graph, is_planar, naslednici_uzlu, predchdci_uzlu, sousedi_uzlu,
    vystupni_okoli_uzlu, vstupni_okoli_uzlu, okoli_uzlu,
    vystupni_stupen_uzlu, vstupni_stupen_uzlu, stupen_uzlu
)

class GraphApp(App):
    CSS = """
    Screen {
        layout: vertical;
    }
    
    #sidebar {
        dock: left;
        width: 30;
        height: 100%;
        background: $panel;
        border-right: vkey $accent;
    }
    
    #content {
        height: 100%;
        width: 1fr;
        padding: 1;
    }
    
    Button {
        width: 100%;
        margin-bottom: 1;
    }
    
    .menu_title {
        text-align: center;
        padding: 1;
        background: $accent;
        color: $text;
        text-style: bold;
        margin-bottom: 1;
    }
    
    RichLog {
        background: $surface;
        color: $text;
        border: solid $accent;
        height: 100%;
        overflow-y: scroll;
    }
    """

    BINDINGS = [
        ("q", "quit", "Ukončit"),
        ("m", "toggle_sidebar", "Přepnout boční panel"),
    ]

    def __init__(self):
        super().__init__()
        self.graph = None
        self.graph_name = "No Graph Loaded"
        self.files = []

    def compose(self) -> ComposeResult:
        yield Header()
        yield Container(
            VerticalScroll(id="sidebar"),
            Container(RichLog(id="output_log", markup=True), id="content"),
        )
        yield Footer()

    def on_mount(self) -> None:
        self.title = "Aplikace Teorie Grafů"
        self.sub_title = self.graph_name
        self.load_files()
        self.show_main_menu()

    def load_files(self):
        base_path = 'graphs'
        if not os.path.exists(base_path):
            os.makedirs(base_path)
        
        self.files = [f for f in os.listdir(base_path) if f.endswith('.tg') or f.endswith('.txt')]
        # Natural sort key implementation
        import re
        def natural_sort_key(name):
            return (len(name), [int(text) if text.isdigit() else text for text in re.split(r'(\d+)', name)])
        self.files.sort(key=natural_sort_key)

    def show_main_menu(self):
        sidebar = self.query_one("#sidebar")
        sidebar.remove_children()
        
        sidebar.mount(Label("HLAVNÍ MENU", classes="menu_title"))
        sidebar.mount(Button("Načíst graf", id="btn_load_graph"))
        
        if self.graph:
            sidebar.mount(Button("Vlastnosti grafu", id="btn_properties"))
            sidebar.mount(Button("Vlastnosti uzlu", id="btn_node_props"))
            sidebar.mount(Button("Matice a algoritmy", id="btn_matrices"))
            sidebar.mount(Button("Toky a řezy", id="btn_flows"))
            sidebar.mount(Button("Průchody grafem", id="btn_traversals"))
        
        sidebar.mount(Button("Ukončit", id="btn_quit", variant="error"))

    def show_file_selection(self):
        sidebar = self.query_one("#sidebar")
        sidebar.remove_children()
        
        sidebar.mount(Label("VYBERTE SOUBOR", classes="menu_title"))
        sidebar.mount(Button("<< Zpět", id="btn_back_main"))
        
        for i, f in enumerate(self.files):
            sidebar.mount(Button(f, id=f"file_idx_{i}"))

    def log_output(self, text):
        log = self.query_one("#output_log", RichLog)
        log.write(text)

    def clear_log(self):
        log = self.query_one("#output_log", RichLog)
        log.clear()

    @on(Button.Pressed)
    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id
        
        if button_id == "btn_quit":
            self.exit()
        elif button_id == "btn_load_graph":
            self.show_file_selection()
        elif button_id == "btn_back_main":
            self.show_main_menu()
        elif button_id and button_id.startswith("file_idx_"):
            idx = int(button_id.replace("file_idx_", ""))
            if 0 <= idx < len(self.files):
                filename = self.files[idx]
                self.load_graph(filename)
        elif button_id == "btn_properties":
            self.show_graph_properties()
        elif button_id == "btn_node_props":
            self.app.push_screen(InputScreen("Zadejte jméno uzlu:", self.show_node_properties))
        elif button_id == "btn_traversals":
            self.show_traversals_menu()
        elif button_id == "btn_matrices":
            self.show_matrices_menu()
        elif button_id == "btn_flows":
             self.show_flows_menu()
        
        # Matrices & Algorithms Categories
        elif button_id == "btn_cat_matrices":
            self.show_matrices_submenu()
        elif button_id == "btn_cat_spanning":
            self.show_spanning_trees_submenu()
        elif button_id == "btn_cat_paths":
            self.show_paths_submenu()

        # Matrix Operations
        elif button_id == "btn_adj_matrix":
            self.run_matrix_op(adjacency_matrix, "Matice sousednosti")
        elif button_id == "btn_inc_matrix":
            self.run_matrix_op(incidence_matrix, "Matice incidence")
        elif button_id == "btn_matrix_pow_2":
            self.run_matrix_pow(2)
        elif button_id == "btn_matrix_pow_3":
            self.run_matrix_pow(3)
        elif button_id == "btn_matrix_pow_n":
            self.app.push_screen(InputScreen("Zadejte mocninu:", self.run_matrix_pow))
        elif button_id == "btn_matrix_laplacian":
            self.run_matrix_op(laplacian_matrix, "Laplaceova matice")
        elif button_id == "btn_matrix_distance":
            self.run_matrix_op(distance_matrix, "Matice délek")
        elif button_id == "btn_matrix_predecessor":
            self.run_matrix_op(predecessor_matrix, "Matice předchůdců")
        elif button_id == "btn_matrix_sign":
            self.run_matrix_op(sign_matrix, "Znaménková matice")
        elif button_id == "btn_matrix_all":
            self.run_all_matrices()

        # Spanning Trees
        elif button_id == "btn_span_count":
            self.run_spanning_tree_op(number_of_spanning_trees, "Počet koster")
        elif button_id == "btn_span_min":
            self.run_spanning_tree_op(minimum_spanning_tree, "Minimální kostra")
        elif button_id == "btn_span_max":
            self.run_spanning_tree_op(maximum_spanning_tree, "Maximální kostra")

        # Paths
        elif button_id == "btn_path_shortest":
            self.app.push_screen(TwoInputScreen("Start:", "Cíl:", lambda s, e: self.run_path_op(get_shortest_path, s, e, "Nejkratší cesta")))
        elif button_id == "btn_path_safest":
            self.app.push_screen(TwoInputScreen("Start:", "Cíl:", lambda s, e: self.run_path_op(
                get_safest_path, s, e, "Nejbezpečnější cesta",
                lambda x: ("Pravděpodobnost úspěchu", f"{x:.2f} %")
            )))
        elif button_id == "btn_path_widest":
            self.app.push_screen(TwoInputScreen("Start:", "Cíl:", lambda s, e: self.run_path_op(get_widest_path, s, e, "Nejširší cesta")))
        elif button_id == "btn_path_longest":
             # Using get_most_dangerous_path which has cycle detection
            self.app.push_screen(TwoInputScreen("Start:", "Cíl:", lambda s, e: self.run_path_op(get_most_dangerous_path, s, e, "Nejdelší cesta")))
        elif button_id == "btn_path_narrowest":
            self.app.push_screen(TwoInputScreen("Start:", "Cíl:", lambda s, e: self.run_path_op(get_narrowest_path, s, e, "Nejužší cesta")))
        elif button_id == "btn_path_dangerous":
            self.app.push_screen(TwoInputScreen("Start:", "Cíl:", lambda s, e: self.run_path_op(get_most_dangerous_path, s, e, "Nejnebezpečnější cesta")))
        elif button_id == "btn_path_safest_prod":
            self.app.push_screen(TwoInputScreen("Start:", "Cíl:", lambda s, e: self.run_path_op(
                get_safest_path_by_boreccz1, s, e, "Nejbezpečnější cesta (max součin)",
                lambda x: ("Pravděpodobnost úspěchu", f"{x:.2f} %")
            )))
        elif button_id == "btn_path_dangerous_prod":
            self.app.push_screen(TwoInputScreen("Start:", "Cíl:", lambda s, e: self.run_path_op(
                get_most_dangerous_path_by_boreccz1, s, e, "Nejnebezpečnější cesta (min součin)",
                lambda x: ("Pravděpodobnost nebezpečí", f"{x:.2f} %" if x != float('inf') else "∞")
            )))
        elif button_id == "btn_path_moore":
            self.app.push_screen(TwoInputScreen("Start:", "Cíl:", lambda s, e: self.run_path_op(moore_shortest_path, s, e, "Moorův algoritmus (BFS)")))
        
        # Traversals
        elif button_id == "btn_dfs":
             self.app.push_screen(InputScreen("Zadejte počáteční uzel:", lambda x: self.run_traversal(dfs, x, "DFS")))
        elif button_id == "btn_bfs":
             self.app.push_screen(InputScreen("Zadejte počáteční uzel:", lambda x: self.run_traversal(bfs_traversal, x, "BFS")))
        elif button_id == "btn_preorder":
             self.app.push_screen(InputScreen("Zadejte počáteční uzel:", lambda x: self.run_traversal(preorder, x, "Preorder")))
        elif button_id == "btn_postorder":
             self.app.push_screen(InputScreen("Zadejte počáteční uzel:", lambda x: self.run_traversal(postorder, x, "Postorder")))
        elif button_id == "btn_inorder":
             self.app.push_screen(InputScreen("Zadejte počáteční uzel:", lambda x: self.run_traversal(inorder, x, "Inorder")))
        elif button_id == "btn_level_order":
             self.app.push_screen(InputScreen("Zadejte počáteční uzel:", lambda x: self.run_traversal(level_order, x, "Level Order")))
        elif button_id == "btn_print_all":
             self.app.push_screen(InputScreen("Zadejte počáteční uzel:", self.run_all_traversals))

        # Flows
        elif button_id == "btn_max_flow":
             self.app.push_screen(TwoInputScreen("Zdroj:", "Cíl:", self.run_max_flow))
        elif button_id == "btn_edmonds_karp":
             self.app.push_screen(TwoInputScreen("Zdroj:", "Cíl:", self.run_edmonds_karp))

    def load_graph(self, filename):
        try:
            file_path = os.path.join('graphs', filename)
            self.graph = load_graph_from_file(file_path)
            self.graph_name = filename
            self.sub_title = filename
            self.clear_log()
            self.log_output(f"[green]Graf {filename} byl úspěšně načten.[/green]")
            self.show_main_menu()
        except Exception as e:
            self.log_output(f"[red]Chyba při načítání grafu: {e}[/red]")

    def run_max_flow(self, source, sink):
        if not self.graph:
            return
        
        self.log_output(f"[bold cyan]Výpočet maximálního toku ({source} -> {sink})[/]")
        try:
            # minimum_cut returns (cut_edges, max_flow_value)
            cut_edges, max_flow = minimum_cut(self.graph, source, sink)
            self.log_output(f"Maximální tok: [bold green]{max_flow}[/]")
            self.log_output(f"Minimální řez (hrany): {', '.join([str(e) for e in cut_edges])}")
        except Exception as e:
            self.log_output(f"[bold red]Chyba: {e}[/]")

    def run_edmonds_karp(self, source, sink):
        if not self.graph:
            return
        
        self.log_output(f"[bold cyan]Edmonds-Karp (Detailní) ({source} -> {sink})[/]")
        try:
            # We need to import edmonds_karp_full locally or ensure it's imported at top
            from test_sec import edmonds_karp_full
            edmonds_karp_full(self.graph, source, sink, logger=self.log_output)
        except Exception as e:
            self.log_output(f"[bold red]Chyba: {e}[/]")

    def show_graph_properties(self):
        if not self.graph: return
        self.clear_log()
        self.log_output("[bold underline yellow]Vlastnosti grafu:[/]")
        
        from properties import (
            is_connected, is_directed, is_tree, is_weighted,
            prosty, jednoduchy, is_finite, is_complete, is_regular, is_bipartite
        )

        def fmt(val):
            return f"[{'bold green' if val else 'bold red'}]{'Ano' if val else 'Ne'}[/]"
        
        self.log_output(f"[cyan]Počet uzlů:[/cyan] [bold white]{len(self.graph.nodes)}[/]")
        self.log_output(f"[cyan]Počet hran:[/cyan] [bold white]{len(self.graph.edges)}[/]")

        self.log_output("[dim]-------------------[/]")
        
        self.log_output(f"[cyan]Ohodnocený:[/cyan] {fmt(is_weighted(self.graph))}")
        self.log_output(f"[cyan]Orientovaný:[/cyan] {fmt(is_directed(self.graph))}")
        self.log_output(f"[cyan]Souvislý:[/cyan] {fmt(is_connected(self.graph))}")
        self.log_output(f"[cyan]Prostý:[/cyan] {fmt(prosty(self.graph))}")
        self.log_output(f"[cyan]Jednoduchý:[/cyan] {fmt(jednoduchy(self.graph))}")
        self.log_output(f"[cyan]Rovinný:[/cyan] {fmt(is_planar(self.graph))}")
        self.log_output(f"[cyan]Konečný:[/cyan] {fmt(is_finite(self.graph))}")
        self.log_output(f"[cyan]Úplný:[/cyan] {fmt(is_complete(self.graph))}")
        self.log_output(f"[cyan]Regulární:[/cyan] {fmt(is_regular(self.graph))}")
        self.log_output(f"[cyan]Bipartitní:[/cyan] {fmt(is_bipartite(self.graph))}")
        self.log_output(f"[cyan]Je strom:[/cyan] {fmt(is_tree(self.graph))}")

    def show_node_properties(self, node_name):
        if not self.graph: return
        self.clear_log()
        if node_name not in self.graph.nodes:
            self.log_output(f"[bold red]Uzel {node_name} neexistuje.[/]")
            return
        
        def czech_format(val):
            if isinstance(val, list):
                # Check if list contains Edges
                if val and hasattr(val[0], 'weight') and hasattr(val[0], 'name'):
                    items = []
                    for e in val:
                        items.append(f"{e.node1} -> {e.node2} (váha: {e.weight}, název: {e.name})")
                    return "[" + ", ".join(items) + "]"
                return str(val)
            return str(val)

        self.log_output(f"[bold underline yellow]Vlastnosti uzlu {node_name}:[/]")
        self.log_output(f"[cyan]Následníci (U+):[/cyan] [white]{naslednici_uzlu(self.graph, node_name)}[/]")
        self.log_output(f"[cyan]Předchůdci (U-):[/cyan] [white]{predchdci_uzlu(self.graph, node_name)}[/]")
        self.log_output(f"[cyan]Sousedé (U):[/cyan] [white]{sousedi_uzlu(self.graph, node_name)}[/]")
        self.log_output(f"[cyan]Výstupní okolí (H+):[/cyan] [white]{czech_format(vystupni_okoli_uzlu(self.graph, node_name))}[/]")
        self.log_output(f"[cyan]Vstupní okolí (H-):[/cyan] [white]{czech_format(vstupni_okoli_uzlu(self.graph, node_name))}[/]")
        self.log_output(f"[cyan]Okolí uzlu:[/cyan] [white]{czech_format(okoli_uzlu(self.graph, node_name))}[/]")
        self.log_output(f"[cyan]Výstupní stupeň (d-):[/cyan] [bold white]{vystupni_stupen_uzlu(self.graph, node_name)}[/]")
        self.log_output(f"[cyan]Vstupni stupeň (d+):[/cyan] [bold white]{vstupni_stupen_uzlu(self.graph, node_name)}[/]")
        self.log_output(f"[cyan]Stupeň uzlu:[/cyan] [bold white]{stupen_uzlu(self.graph, node_name)}[/]")

    def show_traversals_menu(self):
        sidebar = self.query_one("#sidebar")
        sidebar.remove_children()
        sidebar.mount(Label("PRŮCHODY", classes="menu_title"))
        sidebar.mount(Button("<< Zpět", id="btn_back_main"))
        sidebar.mount(Button("DFS", id="btn_dfs"))
        sidebar.mount(Button("BFS", id="btn_bfs"))
        sidebar.mount(Button("Preorder", id="btn_preorder"))
        sidebar.mount(Button("Postorder", id="btn_postorder"))
        sidebar.mount(Button("Inorder", id="btn_inorder"))
        sidebar.mount(Button("Level Order", id="btn_level_order"))
        sidebar.mount(Button("Vypsat vše", id="btn_print_all"))

    def show_flows_menu(self):
        sidebar = self.query_one("#sidebar")
        sidebar.remove_children()
        sidebar.mount(Label("TOKY A ŘEZY", classes="menu_title"))
        sidebar.mount(Button("<< Zpět", id="btn_back_main"))
        sidebar.mount(Button("Max Tok & Min Řez", id="btn_max_flow"))
        sidebar.mount(Button("Edmonds-Karp (Detail)", id="btn_edmonds_karp"))

    def show_matrices_menu(self):
        sidebar = self.query_one("#sidebar")
        sidebar.remove_children()
        sidebar.mount(Label("ALGORITMY", classes="menu_title"))
        sidebar.mount(Button("<< Zpět", id="btn_back_main"))
        sidebar.mount(Button("Matice", id="btn_cat_matrices"))
        sidebar.mount(Button("Kostry", id="btn_cat_spanning"))
        sidebar.mount(Button("Hledání cest", id="btn_cat_paths"))

    def show_matrices_submenu(self):
        sidebar = self.query_one("#sidebar")
        sidebar.remove_children()
        sidebar.mount(Label("MATICE", classes="menu_title"))
        sidebar.mount(Button("<< Zpět", id="btn_matrices"))
        sidebar.mount(Button("Matice sousednosti", id="btn_adj_matrix"))
        sidebar.mount(Button("Matice incidence", id="btn_inc_matrix"))
        sidebar.mount(Button("Mocnina 2", id="btn_matrix_pow_2"))
        sidebar.mount(Button("Mocnina 3", id="btn_matrix_pow_3"))
        sidebar.mount(Button("Mocnina N", id="btn_matrix_pow_n"))
        sidebar.mount(Button("Laplaceova matice", id="btn_matrix_laplacian"))
        sidebar.mount(Button("Matice délek", id="btn_matrix_distance"))
        sidebar.mount(Button("Matice předchůdců", id="btn_matrix_predecessor"))
        sidebar.mount(Button("Znaménková matice", id="btn_matrix_sign"))
        sidebar.mount(Button("Všechny matice", id="btn_matrix_all"))

    def show_spanning_trees_submenu(self):
        sidebar = self.query_one("#sidebar")
        sidebar.remove_children()
        sidebar.mount(Label("KOSTRY", classes="menu_title"))
        sidebar.mount(Button("<< Zpět", id="btn_matrices"))
        sidebar.mount(Button("Počet koster", id="btn_span_count"))
        sidebar.mount(Button("Minimální kostra", id="btn_span_min"))
        sidebar.mount(Button("Maximální kostra", id="btn_span_max"))

    def show_paths_submenu(self):
        sidebar = self.query_one("#sidebar")
        sidebar.remove_children()
        sidebar.mount(Label("CESTY", classes="menu_title"))
        sidebar.mount(Button("<< Zpět", id="btn_matrices"))
        sidebar.mount(Button("Nejkratší", id="btn_path_shortest"))
        sidebar.mount(Button("Nejdelší", id="btn_path_longest"))
        sidebar.mount(Button("Nejbezpečnější (min součet)", id="btn_path_safest"))
        sidebar.mount(Button("Nejbezpečnější (max součin)", id="btn_path_safest_prod"))
        sidebar.mount(Button("Nejnebezpečnější (max součet)", id="btn_path_dangerous"))
        sidebar.mount(Button("Nejnebezpečnější (min součin)", id="btn_path_dangerous_prod"))
        sidebar.mount(Button("Nejširší", id="btn_path_widest"))
        sidebar.mount(Button("Nejužší", id="btn_path_narrowest"))
        sidebar.mount(Button("Moorův alg. (BFS)", id="btn_path_moore"))

    def run_matrix_op(self, func, title, clear=True):
        if not self.graph:
            return
        
        if clear:
            self.clear_log()
        try:
            result = func(self.graph)
            if len(result) == 3:
                matrix, row_labels, col_labels = result
            else:
                matrix, row_labels = result
                col_labels = row_labels
            
            self.log_output(f"\n[bold]{title}:[/]")
            
            # Save to file
            try:
                save_matrix_to_file(matrix, row_labels, col_labels, f"{title.replace(' ', '_')}.csv", title)
                self.log_output(f"[green]💾 Matice uložena do csv_export/{title.replace(' ', '_')}.csv[/]")
            except Exception as e:
                self.log_output(f"[red]Chyba při ukládání do souboru: {e}[/]")

            # Format matrix for display
            table_str = "     " + " ".join([f"{str(l):>4}" for l in col_labels]) + "\n"
            for i, row in enumerate(matrix):
                row_str = f"{str(row_labels[i]):>3}  " + " ".join([f"{str(val):>4}" for val in row])
                table_str += row_str + "\n"
            
            self.log_output(table_str)
            
        except Exception as e:
            self.log_output(f"[red]Chyba při výpočtu {title}: {e}[/red]")

    def run_all_matrices(self):
        if not self.graph: return
        self.clear_log()
        self.log_output("[bold underline yellow]Všechny matice:[/]\n")
        
        matrices = [
            (adjacency_matrix, "Matice sousednosti"),
            (incidence_matrix, "Matice incidence"),
            (laplacian_matrix, "Laplaceova matice"),
            (distance_matrix, "Matice délek"),
            (predecessor_matrix, "Matice předchůdců"),
            (sign_matrix, "Znaménková matice")
        ]
        
        for func, title in matrices:
            self.run_matrix_op(func, title, clear=False)
            self.log_output("\n" + "-"*50 + "\n")

    def run_matrix_pow(self, power):
        if not self.graph: return
        self.clear_log()
        try:
            power = int(power)
            self.log_output(f"[bold underline yellow]Matice sousednosti na {power}:[/]")
            
            # Get adjacency matrix first
            adj_matrix, node_names = adjacency_matrix(self.graph)
            
            # Calculate power
            result_matrix = matrix_power(adj_matrix, power)
            
            # Save to file
            try:
                save_matrix_to_file(result_matrix, node_names, node_names, f"Matice_sousednosti_na_{power}.csv", f"Matice sousednosti na {power}")
                self.log_output(f"[green]💾 Matice uložena do csv_export/Matice_sousednosti_na_{power}.csv[/]")
            except Exception as e:
                self.log_output(f"[red]Chyba při ukládání do souboru: {e}[/]")
            
            # Display matrix
            table_str = "     " + " ".join([f"{str(l):>4}" for l in node_names]) + "\n"
            for i, row in enumerate(result_matrix):
                row_str = f"{str(node_names[i]):>3}  " + " ".join([f"{str(val):>4}" for val in row])
                table_str += row_str + "\n"
            
            self.log_output(table_str)
        except Exception as e:
            self.log_output(f"[bold red]Chyba: {e}[/]")

    def run_spanning_tree_op(self, func, name):
        if not self.graph: return
        self.clear_log()
        self.log_output(f"[bold underline yellow]{name}:[/]")
        try:
            result = func(self.graph)
            
            # Handle tuple return (mst, weight)
            if isinstance(result, tuple):
                mst_edges = result[0]
                weight = result[1]
                self.log_output(f"[cyan]Celková váha:[/cyan] [bold white]{weight}[/]")
                if isinstance(mst_edges, list):
                    for edge in mst_edges:
                        self.log_output(f"[green]{edge.node1}[/] - [green]{edge.node2}[/] ([bold white]{edge.weight}[/])")
                else:
                     self.log_output(f"[dim]{str(mst_edges)}[/]")
            elif isinstance(result, list): # Just edges list
                for edge in result:
                    self.log_output(f"[green]{edge.node1}[/] - [green]{edge.node2}[/] ([bold white]{edge.weight}[/])")
            else:
                self.log_output(f"[bold white]{result}[/]")
        except Exception as e:
            self.log_output(f"[bold red]Chyba: {e}[/]")

    def run_path_op(self, func, start, end, name, format_result=None):
        if not self.graph: return
        self.clear_log()
        
        # Validace vstupů
        if not start or not end or not start.strip() or not end.strip():
            self.log_output(f"[bold underline yellow]{name}:[/]")
            self.log_output("")
            self.log_output("[bold red]❌ Chyba: Musíte zadat oba uzly (start i cíl).[/]")
            return
        
        start = start.strip()
        end = end.strip()
        
        self.log_output(f"[bold underline yellow]{name} z {start} do {end}:[/]")
        self.log_output("")  # Prázdný řádek
        
        try:
            # Path functions usually return (path_nodes, edges, distance) or (path_nodes, edges, distance, has_cycle)
            result = func(self.graph, start, end, verbose=True)  # Zapneme verbose
            
            has_cycle = False
            if isinstance(result, tuple):
                if len(result) == 4:
                    path_nodes, edges, distance, has_cycle = result
                elif len(result) >= 3:
                    path_nodes, edges, distance = result[0], result[1], result[2]
                else:
                    self.log_output(str(result))
                    return

                if distance is None:
                    self.log_output("[bold red]❌ Cesta nebyla nalezena.[/]")
                    return
                
                if distance == float('inf') and not has_cycle:
                    self.log_output("[bold red]❌ Nelze určit: Graf obsahuje kladný cyklus (cesta je nekonečná).[/]")
                    if path_nodes:
                        self.log_output(f"[cyan]Nalezená část cesty (s cyklem):[/cyan] [bold white]{' → '.join(path_nodes)}[/]")
                    return

                # Hlavní výsledek
                self.log_output("[bold yellow]═" * 30 + "[/]")
                self.log_output(f"[bold cyan]VÝSLEDEK: {name}[/]")
                self.log_output("[bold yellow]═" * 30 + "[/]")
                self.log_output("")
                
                # Zobrazení varování o cyklu
                if has_cycle:
                    self.log_output("[bold red]⚠️  VAROVÁNÍ: Graf obsahuje cyklus![/]")
                    self.log_output("[yellow]   Nejdelší cesta může být nekonečně dlouhá.[/]")
                    self.log_output("[yellow]   Zobrazená cesta je nejdelší JEDNODUCHÁ cesta (bez opakování uzlů).[/]")
                    self.log_output("")

                # Metrika / Hodnota
                if format_result:
                    label, formatted_val = format_result(distance)
                    self.log_output(f"📏 [cyan]{label}:[/cyan] [bold white]{formatted_val}[/]")
                else:
                    self.log_output(f"📏 [cyan]Vzdálenost/Cena:[/cyan] [bold white]{distance}[/]")
                
                # Cesta
                path_str = " → ".join([f"[bold green]{n}[/]" for n in path_nodes])
                self.log_output("")
                self.log_output(f"🛤️  [cyan]Cesta ({len(path_nodes)} uzlů):[/cyan]")
                self.log_output(f"   {path_str}")
                
                # Statistiky
                self.log_output("")
                self.log_output("📊 [cyan]Statistiky:[/cyan]")
                self.log_output(f"   • Počet uzlů: [bold white]{len(path_nodes)}[/]")
                self.log_output(f"   • Počet hran: [bold white]{len(edges)}[/]")
                
                # Hrany (pokud není moc)
                if len(edges) <= 10:
                    self.log_output(f"   • Hrany: [dim]{' → '.join([f'{e[0]}-{e[1]}' for e in edges])}[/]")
                
                self.log_output("")
                self.log_output("[bold yellow]═" * 30 + "[/]")
                
            else:
                self.log_output(str(result))
        except ValueError as e:
            # Specifická chyba pro neexistující uzly
            self.log_output(f"[bold red]❌ Chyba: {e}[/]")
            if "not in list" in str(e):
                available_nodes = sorted(self.graph.nodes)
                self.log_output(f"[dim]Dostupné uzly: {', '.join(available_nodes)}[/]")
        except Exception as e:
            self.log_output(f"[bold red]❌ Chyba: {e}[/]")

    def run_traversal(self, func, start_node, name):
        if not self.graph: return
        self.clear_log()
        self.log_output(f"[bold underline yellow]{name} průchod z uzlu {start_node}:[/]")
        try:
            result = func(self.graph, start_node)
            self.log_output(f"[dim]{str(result)}[/]")
        except Exception as e:
            self.log_output(f"[bold red]Chyba: {e}[/]")

    def run_all_traversals(self, start_node):
        if not self.graph: return
        self.clear_log()
        self.log_output(f"[bold underline yellow]Všechny průchody z uzlu {start_node}:[/]")
        self.log_output(f"[cyan]DFS:[/cyan] [dim]{dfs(self.graph, start_node)}[/]")
        self.log_output(f"[cyan]BFS:[/cyan] [dim]{bfs_traversal(self.graph, start_node)}[/]")
        self.log_output(f"[cyan]Preorder:[/cyan] [dim]{preorder(self.graph, start_node)}[/]")
        self.log_output(f"[cyan]Postorder:[/cyan] [dim]{postorder(self.graph, start_node)}[/]")
        self.log_output(f"[cyan]Inorder:[/cyan] [dim]{inorder(self.graph, start_node)}[/]")
        self.log_output(f"[cyan]Level Order:[/cyan] [dim]{level_order(self.graph, start_node)}[/]")

    def run_max_flow(self, source, sink):
        if not self.graph: return
        self.clear_log()
        try:
            cut_edges, max_flow = minimum_cut(self.graph, source, sink)
            self.log_output(f"[bold underline yellow]Maximální tok z {source} do {sink}:[/] [bold white]{max_flow}[/]")
            self.log_output("\n[bold underline yellow]Hrany minimálního řezu:[/]")
            for edge in cut_edges:
                self.log_output(f"[green]{edge.node1}[/] -> [green]{edge.node2}[/] ([cyan]Kapacita:[/cyan] [bold white]{edge.weight}[/])")
        except Exception as e:
            self.log_output(f"[bold red]Chyba: {e}[/]")




class InputScreen(Screen):
    def __init__(self, prompt, callback):
        super().__init__()
        self.prompt = prompt
        self.callback = callback

    def compose(self) -> ComposeResult:
        yield Container(
            Label(self.prompt),
            Input(id="input"),
            Button("OK", id="btn_ok"),
            Button("Zrušit", id="btn_cancel"),
            classes="input_dialog"
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn_ok":
            value = self.query_one("#input", Input).value
            self.dismiss()
            self.callback(value)
        elif event.button.id == "btn_cancel":
            self.dismiss()

class TwoInputScreen(Screen):
    def __init__(self, prompt1, prompt2, callback):
        super().__init__()
        self.prompt1 = prompt1
        self.prompt2 = prompt2
        self.callback = callback

    def compose(self) -> ComposeResult:
        yield Container(
            Label(self.prompt1),
            Input(id="input1"),
            Label(self.prompt2),
            Input(id="input2"),
            Button("OK", id="btn_ok"),
            Button("Zrušit", id="btn_cancel"),
            classes="input_dialog"
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn_ok":
            val1 = self.query_one("#input1", Input).value
            val2 = self.query_one("#input2", Input).value
            self.dismiss()
            self.callback(val1, val2)
        elif event.button.id == "btn_cancel":
            self.dismiss()



if __name__ == "__main__":
    app = GraphApp()
    app.run()

