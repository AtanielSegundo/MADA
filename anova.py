"""
LAST REVISION DATE: 07/04/2025 

Purpose:
    Portuguese:
        O propósito desse script é validar via teste ANOVA a estratégia
        Fast Advanced Pixel utilizando caminho aberto, variando o K
        número de clusters e testar em um grupo de N peças com camadas
        de diferentes geometrias.
        
        Espera-se obter métricas que demonstrem a efetiva redução do
        tempo computacional conforme aumento dos K clusters e espera-se
        verificar como se dá a relação entre o tamanho do percurso e os
        K clusters.
"""
import os 
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'

from typing import Dict, List, Tuple
from pathlib import Path
import argparse
import time

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import f_oneway

from core.TSP.strategy import Strategy
from core.Layer import Layer

VERBOSE = False

def log_info(message: str) -> None:
    if VERBOSE:
        print(f"[INFO] {message}")

class AnovaContext:
    def __init__(self):
        self.data = dict()
        self.args = dict()
        self.isResumed    = False 
        self.current_step = 0
        self.current_part = None
        self.current_kval = None
        self.current_iteration = None

class TestPart:
    """Represents a test part with a layer constructed from a file."""
    def __init__(self, path: str, nick: str = None, **layer_args):
        """
        Args:
            path: File path to load the layer from.
            nick: Short identifier for the part. Defaults to the filename stem.
            layer_args: Additional arguments passed to Layer.From.
        """
        path_obj = Path(path)
        self.id = nick or path_obj.stem
        self.layer = Layer.From(str(path_obj), **layer_args)

class Frame:
    """Stores sample data for a specific part and cluster count."""
    def __init__(self, n_samples: int):
        self.time_s = np.zeros(n_samples)
        self.tour_length = np.zeros(n_samples)
        self.delta_mean = np.zeros(n_samples)
        self.count = 0

    def append_sample(self, time_s: float, tour_length: float, delta_mean: float) -> None:
        """Adds a new sample to the frame."""
        if self.count >= len(self.time_s):
            raise IndexError("Exceeded sample size capacity.")
        self.time_s[self.count] = time_s
        self.tour_length[self.count] = tour_length
        self.delta_mean[self.count] = delta_mean
        self.count += 1

# Parâmetros globais para a solução
SOLVER = "lkh"
GENERATOR = "clusters"
END_TYPE = "open"
HEURISTIC = "continuous"

metrics = ["Computational Time (s)", "Tour Length", "Delta Mean"]
metric_to_attr = {
    "Computational Time (s)": "time_s",
    "Tour Length": "tour_length",
    "Delta Mean": "delta_mean"
}
metric_to_gain = {
    "Computational Time (s)": "Gain Rate of Computational Time",
    "Tour Length": "Gain Rate of Tour Lenght",
    "Delta Mean": "Gain Rate of Delta Mean"
}


def create_data_frame(ctx:AnovaContext,strategy: Strategy, iterations: int, 
                     parts: List[TestPart], k_values: List[int]) -> Dict[str, Dict[int, Frame]]:
    """Generates experimental data by testing different cluster counts on parts."""
    total_steps = len(parts) * len(k_values) * iterations
    if ctx.isResumed:
        i = ctx.current_part
        j = ctx.current_kval
        l = ctx.current_iteration
    else :
        i,j,l = 0 , 0 , 0
    try:
        for ii in range(i,len(parts)):
            log_info(f"Processing Part: {parts[ii].id}")
            print(f"\n[INFO] Processing Part: {parts[ii].id}")
            ctx.data[parts[ii].id] = {}
            ctx.current_part = ii 
            for jj in range(j,len(k_values)):
                log_info(f"Setting K = {k_values[jj]} for {parts[ii].id}")
                print(f"  [INFO] Setting K = {k_values[jj]}")
                ctx.data[parts[ii].id][k_values[jj]] = Frame(iterations)
                strategy.n_cluster = k_values[jj]
                ctx.current_kval = jj
                for ll in range(l,iterations):
                    ctx.current_iteration = ll
                    print(f"    [INFO] Iteration {ll+1}/{iterations} (step {ctx.current_step}/{total_steps})", end='\r', flush=True)
                    # A chamada do solver retorna uma tupla onde o terceiro elemento possui as métricas
                    _, _, metrics_obj = strategy.solve(
                        parts[ii].layer, SOLVER, GENERATOR, 
                        end_type=END_TYPE, initial_heuristic=HEURISTIC
                    )
                    ctx.data[parts[ii].id][k_values[jj]].append_sample(
                        metrics_obj.execution_time,
                        metrics_obj.tour_lenght,  # Considere ajustar para "tour_length" se necessário.
                        metrics_obj.angle_delta_mean
                    )
                    ctx.current_step += 1
                print()  
    except Exception as e:
        handle_exception_save_anova_context(e)
    print("\n[INFO] Data frame creation complete!")
    return ctx.data

def generate_anova_table(data: Dict[str, Dict[int, Frame]], 
                        k_values: List[int]) -> pd.DataFrame:
    """Generates an ANOVA table comparing metrics across cluster configurations."""
    log_info("Generating ANOVA table")
    index_tuples = []
    for metric in metrics:
        for k in k_values:
            index_tuples.append((metric, f"k={k}"))
        index_tuples.append((metric, "p-value"))
    multi_index = pd.MultiIndex.from_tuples(index_tuples, names=["Metric", "Cluster"])
    
    parts = list(data.keys())
    table_rows = []
    
    for metric in metrics:
        attr = metric_to_attr[metric]
        # Para cada valor de K: média e desvio padrão
        for k in k_values:
            row = []
            for part in parts:
                frame = data[part][k]
                values = getattr(frame, attr)[:frame.count]
                mean = np.nanmean(values)
                std = np.nanstd(values)
                row.append(f"{mean:.2f} ± {std:.2f}")
            table_rows.append(row)
        
        # Linha com o p-valor do teste ANOVA para esta métrica, calculado por peça
        pvalue_row = []
        for part in parts:
            groups = [getattr(data[part][k], attr)[:data[part][k].count] 
                      for k in k_values]
            valid_groups = [g for g in groups if len(g) > 0]
            
            if len(valid_groups) < 2:
                p_val = np.nan
            else:
                try:
                    _, p_val = f_oneway(*valid_groups)
                except Exception as e:
                    print(f"[ERROR] ANOVA failed for {part}: {str(e)}")
                    p_val = np.nan
            pvalue_row.append(f"{p_val:.4f}" if not np.isnan(p_val) else "N/A")
        table_rows.append(pvalue_row)
    
    log_info("ANOVA table generated successfully")
    return pd.DataFrame(table_rows, index=multi_index, columns=parts)

def plot_anova_table(df: pd.DataFrame, filename: str = "anova_table.png") -> None:
    """Visualizes the ANOVA table as a styled plot."""
    log_info("Plotting ANOVA table")
    fig, ax = plt.subplots(figsize=(14, df.shape[0]*0.7 + 1))
    ax.axis("off")
    
    table = ax.table(
        cellText=df.values,
        rowLabels=df.index.to_frame().apply(tuple, axis=1),
        colLabels=df.columns,
        cellLoc="center",
        loc="center",
        colColours=["#f0f0f0"]*len(df.columns)
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    log_info(f"ANOVA table plot saved to {filename}")

def dump_anova_table(df: pd.DataFrame, path: str) -> None:
    """Dump the ANOVA table to a text file."""
    log_info(f"Dumping ANOVA table to {path}")
    vals  = df.to_numpy()
    rows_set = list(map(lambda t: t[1], df.index.to_frame().apply(tuple, axis=1)))
    row_set_str = '\',\''.join(rows_set)
    parts = '\',\''.join(df.columns)
    with open(path, "w") as f:
        f.write(f"Metrics: {repr(metrics)}\n")
        f.write(f"Parts: ['{parts}']\n")
        f.write(f"Rows: ['{row_set_str}']\n")
        f.write(str(vals).replace("±", "|"))
    log_info("ANOVA table dumped successfully")

def gen_metrics_graph(anova_path: str, output_dir=None):
    """Generate metrics graphs from a dumped ANOVA table file."""
    log_info(f"Generating metrics graphs from {anova_path}")
    with open(anova_path, "r") as f:
        metrics_list = eval(f.readline().split(":")[-1])
        parts = eval(f.readline().split(":")[-1])
        rows = eval(f.readline().split(":")[-1])
        
        k_set = [v.split('=')[-1] for v in rows[0:(len(rows)//len(metrics_list) - 1)]]
        k_set = list(map(int, k_set))
        k_set_len = len(k_set)

        vals = f.read()
        vals = vals.replace("[", "").replace("]", "").replace("\n", "").strip()
        vals = vals.split(" '")
        vals = [row.split("' '") for row in vals]
        vals = [[elem.strip(" '") for elem in row] for row in vals]
        vals = [[float(elem.split(" | ")[0]) if " | " in elem else float(elem) for elem in row] for row in vals]
        vals = np.array(vals, dtype=float)

        parts_len = len(parts)  
        block_rows = (k_set_len+1)*parts_len
        
        map_to_gain = lambda arr : [arr[0]/v for v in arr] 

        for m, metric in enumerate(metrics_list):
            metric_matrix = np.zeros((parts_len, k_set_len))
            start = m * block_rows
            end = start + block_rows
            block = vals[start:end]
            for i in range(parts_len):
                for j in range(k_set_len):
                    metric_matrix[i][j] = block[j*parts_len + i]
            
            # Metric Graph        
            plt.figure()
            plt.xlabel("Number of Clusters (K)")
            plt.ylabel(metric)
            for i, part in enumerate(parts):
                plt.plot(k_set, metric_matrix[i], label=part)
            plt.legend()
            output_path = os.path.basename(anova_path).replace("table", metric).replace(".txt", ".png")
            if output_dir:
                output_path = os.path.join(output_dir, output_path)
            plt.savefig(output_path)
            plt.close()
            log_info(f"Graph for '{metric}' saved to {output_path}")

            # Gain Rate of Metric Graph
            gain_metric = metric_to_gain[metric]
            plt.figure()
            plt.xlabel("Number of Clusters (K)")
            plt.ylabel(gain_metric)
            for i, part in enumerate(parts):
                plt.plot(k_set, map_to_gain(metric_matrix[i]), label=part)
            plt.legend()
            output_path = os.path.basename(anova_path).replace("table", gain_metric).replace(".txt", ".png")
            if output_dir:
                output_path = os.path.join(output_dir, output_path)
            plt.savefig(output_path)
            plt.close()
            log_info(f"Graph for '{gain_metric}' saved to {output_path}")


def parse_anova_file(anova_path: str) -> Tuple[List[int], List[TestPart]]:
    """Parse the input file to extract the K values and the test parts."""
    log_info(f"Parsing ANOVA file: {anova_path}")
    mode = None  # Pode ser "k" ou "parts"
    k_values: List[int] = []
    parts: List[TestPart] = []
    
    with open(anova_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            lower_line = line.lower()
            if lower_line.startswith("k:"):
                mode = "k"
                line = line.split(":", 1)[1].strip()
                if not line:
                    continue
                log_info("Entered section K")
            elif lower_line.startswith("parts:"):
                mode = "parts"
                line = line.split(":", 1)[1].strip()
                if not line:
                    continue
                log_info("Entered section PARTS")
            
            if mode == "k":
                for token in line.split(","):
                    token = token.strip()
                    if token.isdigit():
                        k_values.append(int(token))
            elif mode == "parts":
                if ":" in line:
                    tag, rest = line.split(":", 1)
                    tag = tag.strip()
                    tokens = [t.strip() for t in rest.split(",") if t.strip()]
                    if tokens:
                        path = tokens[0]
                        kwargs = {}
                        for token in tokens[1:]:
                            if "=" in token:
                                key, value = token.split("=", 1)
                                key = key.strip()
                                value = value.strip()
                                try:
                                    v = int(value)
                                except ValueError:
                                    try:
                                        v = float(value)
                                    except ValueError:
                                        v = value
                                kwargs[key] = v
                        parts.append(TestPart(path, nick=tag, **kwargs))
                        log_info(f"Added TestPart: {tag} -> {path} with args {kwargs}")
    log_info("Parsing complete")
    return k_values, parts         

import pickle

def save_resume_anova_context(ctx: AnovaContext):
    path = f"ANOVA_RESUME_{os.getpid()}.ctx"
    try:
        with open(path, "wb") as f:
            pickle.dump(ctx, f)
        print(f"[INFO] ANOVA context saved to '{path}'")
    except Exception as e:
        print(f"[ERROR] Failed to save ANOVA context: {e}")

def load_resume_anova_context(ctx_path: str) -> AnovaContext:
    try:
        with open(ctx_path, "rb") as f:
            ctx_loaded = pickle.load(f)
        print(f"[INFO] Resuming ANOVA from '{ctx_path}'")
        ctx_loaded.isResumed = True
        return ctx_loaded
    except Exception as e:
        print(f"[ERROR] Failed to load ANOVA context: {e}")
        return AnovaContext()

def handle_exception_save_anova_context(e:Exception):
    print(f"[ERROR] '{e}'")
    log_info("Saving Current ANOVA Session Context")
    save_resume_anova_context(anova_context)
    exit()       

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Anova Table/Graphs Generator")
    parser.add_argument("--source", "-S", nargs="?",
                        help="Generate an ANOVA Table from an ANOVA file")
    parser.add_argument("--graphs", "-G", nargs="?",
                        help="Generate Graphs From An ANOVA Table") 
    parser.add_argument("--output","-O", nargs="?", default="outputs/ANOVA",
                        help="Output Folder to put the processing results") 
    parser.add_argument("--seed", nargs="?", default=time.time_ns() // 4294967295,
                        help="Seed used for deterministic output") 
    parser.add_argument("--distance", nargs="?", default=5, type=int,
                        help="Resolution used to rasterize the part") 
    parser.add_argument("--iterations", "-I", nargs="?", default=50, type=int,
                        help="Iterations used in ANOVA sampling") 
    parser.add_argument("--solver_runs", "-R", nargs="?", default=1, type=int,
                        help="Number of runs used by the solver") 
    parser.add_argument("--heuristic", "-H", nargs="?", default=HEURISTIC,
                        help="Heuristic to use in path planning") 
    parser.add_argument("--verbose", "-V", action="store_true",
                        help="Enable verbose logging")
    parser.add_argument("--resume", "-F", nargs="?", default=None,
                        help="Resume ANOVA test from an existing state") 
    
    anova_context = AnovaContext()

    args = parser.parse_args()
    VERBOSE = args.verbose

    import signal,sys

    def on_sigint(signum, frame):
        handle_exception_save_anova_context(KeyboardInterrupt())
        sys.exit(1)

    signal.signal(signal.SIGINT, on_sigint)

    if args.resume is None:
        anova_context.args["sampling_iterations"] = args.iterations 
        anova_context.args["output"]      = args.output
        anova_context.args["source"]      = args.source
        anova_context.args["solver_runs"] = args.solver_runs 
        anova_context.args["seed"]        = args.seed 
        anova_context.args["distance"]    = args.distance 
        anova_context.args["heuristic"]   = args.heuristic
    else:
        if os.path.exists(args.resume):
            anova_context = load_resume_anova_context(args.resume)
            log_info("Anova Context Loaded From Previous Session")
            log_info("Anova Context Struct:")
            if VERBOSE:
                print("\n".join([f" "*4+f"{k} = {v}" for k,v in anova_context.args.items()]))
                print(" "*4 + f"isResumed = {anova_context.isResumed}")
                print(" "*4 + f"current_step = {anova_context.current_step}")
                print(" "*4 + f"current_part = {anova_context.current_part}")
                print(" "*4 + f"current_kval = {anova_context.current_kval}")
                print(" "*4 + f"current_iteration = {anova_context.current_iteration}")
        else:
            print(f"[ERROR] '{os.path.relpath(args.resume)}' Not Found")

    os.makedirs(anova_context.args["output"], exist_ok=True)

    if args.graphs:
        table_path = args.graphs 
        if os.path.exists(table_path):
            try:
                gen_metrics_graph(table_path, anova_context.args["output"])
            except Exception as e:
                print(f"[ERROR] Parsing went wrong - {e}")
        else:
            print(f"[ERROR] '{os.path.relpath(table_path)}' Not Found")
        exit(1)
    
    log_info("Initializing strategy")
    strategy = Strategy(distance=anova_context.args["distance"], 
                        seed=anova_context.args["seed"], 
                        runs=anova_context.args["solver_runs"])
    
    log_info(f"Parsing source file: {anova_context.args['source']}")
    k_values, parts = parse_anova_file(anova_context.args["source"])
    
    log_info("Creating experimental data frame")
    experimental_data = create_data_frame(anova_context,
                                        strategy, 
                                        anova_context.args["sampling_iterations"], 
                                        parts, 
                                        k_values)
    
    log_info("Generating ANOVA table from experimental data")
    anova_df = generate_anova_table(experimental_data, k_values)
    
    heuristic = anova_context.args["heuristic"]
    dump_path = os.path.join(anova_context.args["output"], f"anova_table_{heuristic}.txt")
    dump_anova_table(anova_df, dump_path)
    plot_path = os.path.join(anova_context.args["output"], f"anova_results_{heuristic}.png")
    plot_anova_table(anova_df, filename=plot_path)
    log_info("Process completed successfully!")