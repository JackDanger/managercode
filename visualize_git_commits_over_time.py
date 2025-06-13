import subprocess
import sys
import os
import argparse
from collections import Counter, defaultdict
from datetime import datetime, date
import calendar
from datetime import timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from scipy.interpolate import make_interp_spline
import re
import json
import hashlib
import time

def run_git_command(cmd, cwd):
    """Run a git command in a specific directory and return output lines."""
    try:
        result = subprocess.check_output(cmd, cwd=cwd, stderr=subprocess.DEVNULL).decode('utf-8')
        return result.splitlines()
    except subprocess.CalledProcessError:
        print(f"Error running command in {cwd}: {' '.join(cmd)}")
        return []

def get_filtered_commit_dates(path):
    """
    Extract YYYY-MM commit dates from the git log, filtering out merge commits and bot commits.
    """
    # Extract hashes of merge commits
    merge_hashes = set(run_git_command(['git', 'rev-list', '--merges', 'HEAD'], cwd=path))

    # Extract commit hash and message for all commits
    log_output = run_git_command(
        ['git', 'log', '--date=format:%Y-%m', '--pretty=format:%H|%ad|%s', '.'],
        cwd=path
    )

    month_counts = Counter()
    for line in log_output:
        try:
            commit_hash, date, message = line.split('|', 2)
        except ValueError:
            continue  # skip malformed lines

        if commit_hash in merge_hashes:
            continue
        if re.match(r'^Merge( branch| pull request)', message):
            continue
        if 'bot' in message.lower():
            continue

        month_counts[date] += 1

    return month_counts

def prorate_current_month(commit_data):
    """Pro-rate the current month's commit count based on how far through the month we are."""
    today = date.today()
    current_month = today.strftime("%Y-%m")
    
    if current_month in commit_data:
        # Calculate pro-rating factor
        days_in_month = calendar.monthrange(today.year, today.month)[1]
        days_elapsed = today.day
        
        # Pro-rate: scale up based on how much of the month remains
        prorate_factor = days_in_month / days_elapsed
        
        # Apply pro-rating to current month
        commit_data[current_month] = int(commit_data[current_month] * prorate_factor)
    
    return commit_data

def normalize_commit_series(commit_data, all_months):
    """Fills in 0s for months without commits."""
    return [commit_data.get(month, 0) for month in all_months]

def get_cache_key(directories):
    """Generate a cache key based on the directories being processed."""
    # Create a consistent hash from the sorted absolute paths
    abs_paths = [os.path.abspath(d) for d in directories]
    abs_paths.sort()
    cache_string = "|".join(abs_paths)
    return hashlib.md5(cache_string.encode()).hexdigest()

def get_cache_filename(cache_key):
    """Get the cache filename for a given cache key."""
    return f"git_commits_cache_{cache_key}.json"

def load_cached_data(cache_key):
    """Load cached data if it exists and is recent."""
    cache_file = get_cache_filename(cache_key)
    if not os.path.exists(cache_file):
        return None
    
    try:
        with open(cache_file, 'r') as f:
            cached_data = json.load(f)
        
        # Check if cache is less than 1 hour old
        cache_time = cached_data.get('timestamp', 0)
        if time.time() - cache_time > 3600:  # 1 hour
            print(f"Cache file {cache_file} is older than 1 hour, regenerating...")
            return None
            
        print(f"Loaded cached data from {cache_file}")
        return cached_data
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Error reading cache file {cache_file}: {e}")
        return None

def save_cached_data(cache_key, repo_commit_data, all_months_set):
    """Save the calculated data to cache."""
    cache_file = get_cache_filename(cache_key)
    cache_data = {
        'timestamp': time.time(),
        'repo_commit_data': repo_commit_data,
        'all_months_set': list(all_months_set)
    }
    
    try:
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f, indent=2)
        print(f"Saved cached data to {cache_file}")
    except Exception as e:
        print(f"Error saving cache file {cache_file}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Plot commit activity over time for directories inside git repos.")
    parser.add_argument("dirs", nargs='+', help="Directories inside git repositories")
    parser.add_argument("--output", default="commit_activity.png", help="Output PNG filename")
    parser.add_argument("--no-cache", action="store_true", help="Skip cache and regenerate data")
    args = parser.parse_args()

    cache_key = get_cache_key(args.dirs)
    dir_basenames = [os.path.basename(dir) for dir in args.dirs]
    
    # Try to load from cache first
    cached_data = None if args.no_cache else load_cached_data(cache_key)
    
    if cached_data:
        # Use cached data
        repo_commit_data = cached_data['repo_commit_data']
        all_months_set = set(cached_data['all_months_set'])
        print("Using cached git data")
    else:
        # Generate data from git commands
        print("Calculating git data...")
        all_months_set = set()
        repo_commit_data = {}

        for directory in args.dirs:
            abs_path = os.path.abspath(directory)
            if not os.path.exists(os.path.join(abs_path, '.')):  # just basic check
                print(f"Invalid path: {abs_path}")
                continue
            commit_counts = get_filtered_commit_dates(abs_path)
            # Pro-rate current month before storing
            commit_counts = prorate_current_month(commit_counts)
            repo_commit_data[abs_path] = commit_counts
            all_months_set.update(commit_counts.keys())
        
        # Save to cache
        save_cached_data(cache_key, repo_commit_data, all_months_set)

    all_months = sorted(all_months_set)
    all_months_dt = [datetime.strptime(m, "%Y-%m") for m in all_months]

    # Calculate peak values and sort repositories by peak commit count (highest to lowest)
    repo_peaks = []
    for path, counts in repo_commit_data.items():
        normalized = normalize_commit_series(counts, all_months)
        peak_value = max(normalized) if normalized else 0
        repo_peaks.append((path, counts, peak_value))
    
    # Sort by peak value in descending order
    repo_peaks.sort(key=lambda x: x[2], reverse=True)
    
    # Prepare data for stacked area plot
    repo_data = []
    repo_labels = []
    for path, counts, peak_value in repo_peaks:
        label = os.path.basename(os.path.normpath(path))
        normalized = normalize_commit_series(counts, all_months)
        repo_data.append(normalized)
        repo_labels.append(label)
    
    # Create smooth curves using interpolation
    x_smooth = np.linspace(0, len(all_months_dt)-1, len(all_months_dt)*2)  # 2x more points for smoothness
    all_months_dt_smooth = [all_months_dt[0] + (all_months_dt[-1] - all_months_dt[0]) * (i / (len(x_smooth)-1)) for i in range(len(x_smooth))]
    
    repo_data_smooth = []
    for data in repo_data:
        if len(data) > 3:  # Need at least 4 points for cubic spline
            # Convert to numpy array for interpolation
            x_orig = np.arange(len(data))
            y_orig = np.array(data)
            
            # Create smooth interpolation
            spline = make_interp_spline(x_orig, y_orig, k=min(2, len(data)-1))
            y_smooth = spline(x_smooth)
            
            # Ensure no negative values (commits can't be negative)
            y_smooth = np.maximum(y_smooth, 0)
            repo_data_smooth.append(y_smooth)
        else:
            # For datasets with few points, just repeat the pattern
            repo_data_smooth.append(np.interp(x_smooth, np.arange(len(data)), data))
    
    # Plotting with stacked area
    plt.figure(figsize=(15, 7))  # Much wider figure to accommodate legend on the left
    plt.stackplot(all_months_dt_smooth, *repo_data_smooth, labels=repo_labels, alpha=0.8)

    # Set x-axis limits to match actual data range, with 5 days padding on each end
    plt.xlim(all_months_dt[0] - timedelta(days=5), all_months_dt[-1] + timedelta(days=15))
    
    plt.xlabel("Month")
    plt.ylabel("Total Non-Merge Commits (Stacked)")
    plt.title("Monthly Commits")
    plt.legend(bbox_to_anchor=(-0.1, 1), loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.subplots_adjust(left=0.25)  # Add extra space on the left for the legend
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=max(len(all_months)//12, 1)))
    plt.xticks(rotation=45)

    plt.savefig(args.output, dpi=200)
    print(f"Saved chart to {args.output}")

if __name__ == "__main__":
    main()
