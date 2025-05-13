#!/usr/bin/env python3
"""
repo_groups.py

Usage:
    python repo_groups.py \
        --group ~/www/some-org-*="Some Org" ~/www/repo1="App 1" ~/www/repo2="Prod App" \
        --group "Infra" ~/www/repo3="Product 4"

This script parses groups of git repositories, computes commit counts and total
lines of tracked files in each repo, then uses log(commits)+log(lines) to size
each group's box. Repos are drawn as circles sized by commit count; group boxes
have a minimum size and are roughly square.
"""
import sys, os, glob, subprocess
import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    """
    Parses:
      --group "Group Name" path1[=Display1] ...
    Supports globbing. Returns:
      [{ 'name': str, 'repos': [ { 'path': abs_path, 'name': display }, ... ] }, ...]
    """
    groups, args, i = [], sys.argv[1:], 0
    while i < len(args):
        if args[i] in ("-g", "--group"):
            if i + 1 >= len(args):
                sys.exit("Error: --group needs a name")
            grp_name = args[i + 1].split("=")[1]
            i += 2
            repos = []
            while i < len(args) and args[i] not in ("-g", "--group"):
                entry = args[i]
                i += 1
                if "=" in entry:
                    pat, disp = entry.split("=", 1)
                else:
                    pat, disp = entry, None
                pat = os.path.abspath(os.path.expanduser(pat))
                matches = glob.glob(pat) or [pat]
                for m in matches:
                    display = disp or os.path.basename(m)
                    repos.append({"path": m, "name": display})
            groups.append({"name": grp_name, "repos": repos})
        else:
            sys.exit(f"Unknown arg: {args[i]}")
    if not groups:
        sys.exit("Error: need at least one --group")
    return groups


def get_commit_count(target_path):
    """Return number of commits in this directory (or full repo)."""
    prev = os.getcwd()
    try:
        os.chdir(target_path)
        proc = subprocess.run(
            "git log --oneline . | wc -l",
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=True,
        )
        return int(proc.stdout.strip())
    finally:
        os.chdir(prev)


def get_total_lines(target_path):
    """Return total lines across all tracked files in repo_path."""
    os.chdir(target_path)
    res = subprocess.run(
        ["git", "ls-files", "."],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=True,
    )
    total = 0
    for f in res.stdout.splitlines():
        p = os.path.join(target_path, f)
        try:
            with open(p, "rb") as fh:
                total += sum(1 for line in fh if line.rstrip(b'\r\n'))
        except:
            pass
    return total


def collect_metrics(groups):
    """Compute commit & line counts per repo."""
    repo_metrics = []
    for grp in groups:
        grp_data = []
        for r in grp["repos"]:
            c = get_commit_count(r["path"])
            l = get_total_lines(r["path"])
            grp_data.append({"commit_count": c, "line_count": l})
        repo_metrics.append(grp_data)
    return repo_metrics


def plot(groups, repo_metrics):
    fig, ax = plt.subplots(figsize=(len(groups)*4, 8))
    ax.axis('off')
    x = np.arange(len(groups))

    # 1) draw repo circles
    xs, ys, commit_sizes, line_sizes, labels = [], [], [], [], []
    for gi, grp in enumerate(groups):
        n = len(grp['repos'])
        for ri, repo in enumerate(grp['repos']):
            c = repo_metrics[gi][ri]['commit_count']
            l = repo_metrics[gi][ri]['line_count']
            xs.append(x[gi])
            ys.append((ri+1)/(n+1))
            # Use log scale for both metrics
            commit_sizes.append(c + 1)
            line_sizes.append(l + 1)
            labels.append((repo['name'], c, l))

    # Convert to arrays and normalize
    commit_sizes = np.array(commit_sizes, float)
    line_sizes = np.array(line_sizes, float)
    
    if commit_sizes.size:
        commit_areas = (commit_sizes) / 50
        line_areas = (line_sizes) / 700
    else:
        commit_areas = np.array([])
        line_areas = np.array([])

    # Draw circles - larger one first (lower z-index)
    for xx, yy, commit_area, line_area in zip(xs, ys, commit_areas, line_areas):
        # Determine which is larger to set z-order
        if commit_area > line_area:
            # Commits circle (blue) is larger, draw first
            ax.scatter(xx, yy, s=commit_area, color='blue', alpha=0.4, edgecolor='k', zorder=1)
            ax.scatter(xx, yy, s=line_area, color='red', alpha=0.4, edgecolor='k', zorder=2)
        else:
            # Lines circle (red) is larger, draw first
            ax.scatter(xx, yy, s=line_area, color='red', alpha=0.4, edgecolor='k', zorder=1)
            ax.scatter(xx, yy, s=commit_area, color='blue', alpha=0.4, edgecolor='k', zorder=2)
    
    # Draw labels with metrics
    for xx, yy, (lbl, commits, lines) in zip(xs, ys, labels):
        # Main label
        ax.text(xx, yy + 0.02, lbl, ha='center', va='bottom', fontsize=10, zorder=3)
        # Metrics in smaller font
        metrics = f"{commits:,} commits, {lines:,} lines"
        ax.text(xx, yy - 0.02, metrics, ha='center', va='top', fontsize=7, zorder=3)

    # 2) compute mass = log(commits+1)+log(lines+1)
    masses = []
    for grp_data in repo_metrics:
        tc = sum(d['commit_count'] for d in grp_data)
        tl = sum(d.get('line_count', 0)   for d in grp_data)
        masses.append(np.log(tc+1) + np.log(tl+1))
    max_mass = max(masses) if masses and max(masses)>0 else 1.0

    # 3) draw group rectangles (width ∝ mass, height fits repos)
    width_base = 0.8
    min_width = 0.3
    for gi, grp in enumerate(groups):
        n = len(grp['repos'])
        if n == 0:
            continue

        # height so that circles (at y=(ri+1)/(n+1)) fit nicely
        y0 = 1/(n+1) - 0.08  # Increased spacing to accommodate metrics text
        height = (n/(n+1) + 0.08) - y0  # Increased spacing to accommodate metrics text

        # width directly from mass
        raw_w = (masses[gi] / max_mass) * width_base
        width = max(raw_w, min_width)

        x0 = x[gi] - width/2
        rect = plt.Rectangle((x0, y0), width, height,
                             fill=False, linewidth=1.5)
        ax.add_patch(rect)

        # only show group name
        ax.text(x[gi], y0 + height + 0.02,
                grp['name'],
                ha='center', va='bottom', weight='bold', fontsize=10)

    ax.set_xlim(-0.5, len(groups)-0.5)
    ax.set_ylim(0, 1.2)
    plt.tight_layout()
    plt.show()


def main():
    groups = parse_args()
    repo_metrics = collect_metrics(groups)
    plot(groups, repo_metrics)


if __name__ == "__main__":
    main()
