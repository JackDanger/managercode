#!/usr/bin/env python3
"""
repo_groups.py

Usage:
    python repo_groups.py \
        --group ~/www/some-org-*="Some Org" ~/www/repo1="App 1" ~/www/repo2="Prod App" \
        --group "Infra" ~/www/repo3="Product 4"

This script parses groups of git repositories, computes commit counts and total lines of tracked files, and renders a minimalist visualization where repo circle areas correspond to
commit counts. Groups are outlined with their names.
"""
import sys
import os
import glob
import subprocess
import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    """
    Parses:
      --group "Group Name" path1[=Display1] path2[=Display2] ...
    Supports globbing in each path.
    Returns list of groups:
      [{ 'name': str,
         'repos': [ { 'path': abs_path, 'name': display }, … ]
      , …}]
    """
    groups = []
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] in ("-g", "--group"):
            if i + 1 >= len(args):
                sys.exit("Error: --group needs a name")
            grp_name = args[i + 1].split('=')[1]
            i += 2
            repos = []
            while i < len(args) and args[i] not in ("-g", "--group"):
                entry = args[i]
                if "=" in entry:
                    pat, disp = entry.split("=", 1)
                else:
                    pat, disp = entry, None
                # expand home/relative
                pat = os.path.abspath(os.path.expanduser(pat))
                matches = glob.glob(pat)
                # if glob yields nothing, treat literal
                if not matches:
                    matches = [pat]
                for m in matches:
                    display = disp or os.path.basename(m)
                    repos.append({"path": m, "name": display})
                i += 1
            groups.append({"name": grp_name, "repos": repos})
        else:
            sys.exit(f"Unknown arg: {args[i]}")
    if not groups:
        sys.exit("Error: need at least one --group")
    return groups


def collect_metrics(groups, get_commit_count):
    """
    For each group, compute per-repo metrics _and_ group total commits.
    Returns two parallel lists:
      repo_metrics = [[{commit_count, line_count},…], …]
      group_commits = [sum_of_commits, …]
    """
    repo_metrics = []
    group_commits = []
    for grp in groups:
        met = []
        total = 0
        for r in grp["repos"]:
            c = get_commit_count(r["path"])
            met.append({"commit_count": c})
            total += c
        repo_metrics.append(met)
        group_commits.append(total)
    return repo_metrics, group_commits


def plot(groups, repo_metrics, group_commits):
    fig, ax = plt.subplots(figsize=(len(groups) * 2, 4))
    ax.axis("off")
    x_pos = np.arange(len(groups))
    # flatten repos for circles
    xs, ys, sz, lbl = [], [], [], []
    for gi, grp in enumerate(groups):
        n = len(grp["repos"])
        for ri, repo in enumerate(grp["repos"]):
            c = repo_metrics[gi][ri]["commit_count"]
            xs.append(x_pos[gi])
            ys.append((ri + 1) / (n + 1))
            sz.append(c)
            lbl.append(f"{repo['name']}")
    sz = np.array(sz, float)
    max_sz = sz.max() if sz.size else 1
    areas = sz / max_sz * 2000
    ax.scatter(xs, ys, s=areas, alpha=0.6, edgecolor="k")
    for x, y, t in zip(xs, ys, lbl):
        ax.text(x, y, t, ha="center", va="center", fontsize=8)
    # now draw group boxes with width ∝ total commits
    max_grp = max(group_commits) or 1
    width_base = 0.8
    for gi, grp in enumerate(groups):
        n = len(grp["repos"])
        if n == 0:
            continue
        total = group_commits[gi]
        w = (total / max_grp) * width_base
        h0 = 1 / (n + 1) - 0.05
        h1 = n / (n + 1) + 0.05 - h0
        x0 = x_pos[gi] - w / 2
        rect = plt.Rectangle((x0, h0), w, h1, fill=False, lw=1.5)
        ax.add_patch(rect)
        ax.text(
            x_pos[gi],
            h0 + h1 + 0.02,
            f"{grp['name']} ({total} commits)",
            ha="center",
            va="bottom",
            weight="bold",
        )
    ax.set_xlim(-0.5, len(groups) - 0.5)
    ax.set_ylim(0, 1.2)
    plt.tight_layout()
    plt.show()


def get_commit_count(target_path):
    """Return the number of commits in the specified directory (or entire repo if it’s the root)."""
    # Save current working directory so we can restore it later
    prev_cwd = os.getcwd()
    try:
        # Change into the target directory
        os.chdir(target_path)
        # Run `git log --oneline | wc -l` to count commits
        proc = subprocess.run(
            "git log --oneline | wc -l",
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=True,
        )
        return int(proc.stdout.strip())
    except Exception:
        return 0
    finally:
        # Restore original working directory
        os.chdir(prev_cwd)


def get_total_lines(repo_path):
    """Return total lines across all files tracked by git."""
    try:
        res = subprocess.run(
            ["git", "-C", repo_path, "ls-files"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=True,
        )
        files = res.stdout.strip().splitlines()
        total = 0
        for f in files:
            full = os.path.join(repo_path, f)
            try:
                with open(full, "rb") as fh:
                    total += sum(1 for _ in fh)
            except Exception:
                pass
        return total
    except Exception:
        return 0


def main():
    groups = parse_args()
    # assume get_commit_count is already defined as per previous snippet
    repo_metrics, group_commits = collect_metrics(groups, get_commit_count)
    plot(groups, repo_metrics, group_commits)


if __name__ == "__main__":
    main()
