#!/usr/bin/env python3
"""
Utility to analyze saved PBS run logs and generate summary plots for successful games.

Parses log files in run_logs/run_*.log, extracts metrics, and saves a multi-panel plot.
"""

import glob
import os
import re
import statistics
from typing import List, Dict


RUN_SOLVED_PATTERN = re.compile(r"Run\s+(\d+):\s+SOLVED")
RUN_FAIL_PATTERN = re.compile(r"Run\s+(\d+):\s+(NO SOLUTION FOUND|TIMED OUT.*)")
NODES_GEN_PATTERN = re.compile(r"Nodes Generated:\s+(\d+)")
NODES_EXP_PATTERN = re.compile(r"Nodes Expanded:\s+(\d+)")
AGENT_STEPS_PATTERN = re.compile(r"Agent\s+(\d+):\s+(\d+)\s+steps")


def parse_log(filepath: str, max_steps_limit: int) -> Dict:
    solved = False
    timed_out = False
    nodes_generated = None
    nodes_expanded = None
    agent_steps = {}

    with open(filepath, "r") as f:
        for raw_line in f:
            line = raw_line.strip()

            match_gen = NODES_GEN_PATTERN.search(line)
            if match_gen:
                nodes_generated = int(match_gen.group(1))

            match_exp = NODES_EXP_PATTERN.search(line)
            if match_exp:
                nodes_expanded = int(match_exp.group(1))

            if "TIMED OUT" in line:
                solved = False
                timed_out = True

            run_fail = RUN_FAIL_PATTERN.search(line)
            if run_fail:
                solved = False

            run_match = RUN_SOLVED_PATTERN.search(line)
            if run_match:
                solved = True

            if solved:
                match_agent = AGENT_STEPS_PATTERN.search(line)
                if match_agent:
                    agent_id = int(match_agent.group(1))
                    steps = int(match_agent.group(2))
                    agent_steps[agent_id] = steps

    max_steps = max(agent_steps.values()) if agent_steps else None
    avg_steps = statistics.mean(list(agent_steps.values())) if agent_steps else None
    total_agents = 4
    success_agents = 0
    over_limit_agents = 0
    if agent_steps:
        success_agents = sum(1 for steps in agent_steps.values() if steps <= max_steps_limit)
        over_limit_agents = sum(1 for steps in agent_steps.values() if steps > max_steps_limit)
    success_ratio = success_agents / total_agents if solved else 0.0

    return {
        "file": os.path.basename(filepath),
        "solved": solved,
        "timed_out": timed_out,
        "success_ratio": success_ratio,
        "max_steps": max_steps,
        "avg_steps": avg_steps,
        "nodes_generated": nodes_generated,
        "nodes_expanded": nodes_expanded,
        "over_limit_agents": over_limit_agents,
        "success_agents": success_agents,
        "total_agents": total_agents,
    }


def main():
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError:
        plt = None

    max_steps_limit = 100
    log_files = sorted(glob.glob(os.path.join("run_logs", "run_*.log")))
    records: List[Dict] = []
    for path in log_files:
        rec = parse_log(path, max_steps_limit)
        if rec:
            records.append(rec)

    if not records:
        print("No runs found in run_logs.")
        return

    runs = list(range(1, len(records) + 1))
    success_pct = [r["success_ratio"] * 100 for r in records]
    max_steps = [r["max_steps"] for r in records]
    avg_steps = [r["avg_steps"] for r in records]
    nodes_gen = [r["nodes_generated"] for r in records]
    nodes_exp = [r["nodes_expanded"] for r in records]
    timed_out = [1 if r["timed_out"] else 0 for r in records]
    over_limit_agents = [r.get("over_limit_agents", 0) for r in records]
    success_agents = [r.get("success_agents", 0) for r in records]

    def safe_mean(values):
        vals = [v for v in values if v is not None]
        return statistics.mean(vals) if vals else float("nan")

    print(f"Parsed {len(records)} runs.")
    solved_runs = sum(1 for r in records if r["solved"])
    total_runs = len(records)
    total_success_agents = sum(success_agents)
    total_agents = sum(r.get("total_agents", 4) for r in records)
    print(f"Solved runs: {solved_runs}")
    print(f"Timed-out runs: {sum(timed_out)}")
    print(f"Agents exceeding {max_steps_limit} steps: {sum(over_limit_agents)}")
    print(f"Average agent success: {(total_success_agents / total_agents) * 100:.2f}%")
    print(f"Average max steps (solved only): {safe_mean(max_steps):.2f}")
    print(f"Average mean steps (solved only): {safe_mean(avg_steps):.2f}")
    print(f"Average nodes generated: {safe_mean(nodes_gen):.2f}")
    print(f"Average nodes expanded: {safe_mean(nodes_exp):.2f}")

    if plt is None:
        print("matplotlib is not installed. Install it to generate plots (pip install matplotlib).")
        return

    fig, axes = plt.subplots(3, 2, figsize=(12, 10))

    axes[0, 0].plot(runs, success_pct, marker="o")
    axes[0, 0].set_title("Agent Success Percentage")
    axes[0, 0].set_xlabel("Successful Run Index")
    axes[0, 0].set_ylabel("% Success")
    axes[0, 0].set_ylim(0, 100)

    plot_max_steps = [v if v is not None else float("nan") for v in max_steps]
    axes[0, 1].plot(runs, plot_max_steps, marker="o", color="orange")
    axes[0, 1].set_title("Max Steps per Successful Run")
    axes[0, 1].set_xlabel("Successful Run Index")
    axes[0, 1].set_ylabel("Steps")

    plot_avg_steps = [v if v is not None else float("nan") for v in avg_steps]
    axes[1, 0].plot(runs, plot_avg_steps, marker="o", color="green")
    axes[1, 0].set_title("Average Steps per Successful Run")
    axes[1, 0].set_xlabel("Successful Run Index")
    axes[1, 0].set_ylabel("Steps")

    plot_nodes_gen = [v if v is not None else float("nan") for v in nodes_gen]
    axes[1, 1].plot(runs, plot_nodes_gen, marker="o", color="purple")
    axes[1, 1].set_title("Nodes Generated")
    axes[1, 1].set_xlabel("Successful Run Index")
    axes[1, 1].set_ylabel("Count")

    plot_nodes_exp = [v if v is not None else float("nan") for v in nodes_exp]
    axes[2, 0].plot(runs, plot_nodes_exp, marker="o", color="red")
    axes[2, 0].set_title("Nodes Expanded")
    axes[2, 0].set_xlabel("Successful Run Index")
    axes[2, 0].set_ylabel("Count")

    axes[2, 1].bar(runs, timed_out, color="red", alpha=0.6, label="Timed Out")
    axes[2, 1].bar(runs, over_limit_agents, color="orange", alpha=0.6, label="Agents >100")
    axes[2, 1].set_title("Timeout / Over-limit Agents")
    axes[2, 1].set_xlabel("Run")
    axes[2, 1].set_ylabel("Indicator")
    axes[2, 1].legend()

    plt.tight_layout()
    output_path = "log_metrics.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved metrics plot to {output_path}")


if __name__ == "__main__":
    main()
