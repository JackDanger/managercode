#!/usr/bin/env python3
"""
A simple command-line application that traverses a directory, counts the lines in each file,
and prints the total line counts summed by file extension.
"""
import argparse
import os
import glob
import sys
import subprocess
from collections import defaultdict

# Set of valid file extensions to count
VALID_EXTENSIONS = {
    'java', 'py', 'cs', 'php', 'xml', 'yaml', 'yml', 'xslt', 'html', 'htm',
    'js', 'ts', 'tsx', 'sh', 'vb', 'asp', 'xls', 'c', 'cpp', 'cxx', 'cc',
    'h', 'hpp', 'hxx', 'go', 'rb', 'swift', 'kt', 'kts', 'pl', 'pm', 'ps1',
    'bat', 'vbs', 'asm', 's', 'cob', 'cbl', 'for', 'f90', 'ada', 'pas',
    'lisp', 'lsp', 'scm', 'clj', 'cljs', 'scala', 'groovy', 'dart', 'm',
    'vhd', 'vhdl', 'v', 'r', 'erl', 'ex', 'exs', 'hs', 'lhs', 'ml',
    'css', 'scss', 'sass', 'less'
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Count lines of code in files by extension in a directory."
    )
    parser.add_argument("directory", help="Path to the directory to scan")
    return parser.parse_args()


def count_lines_by_extension(root_dir):
    """
    Use git ls-files to get tracked files, count lines in each file,
    and return a dict mapping file extensions to total line counts.
    Only counts files with extensions in VALID_EXTENSIONS.
    """
    ext_counts = defaultdict(int)

    try:
        # Run git ls-files and get the output
        if os.path.isdir(os.path.join(root_dir, '.git')):
            result = subprocess.run(
                ['git', 'ls-files'],
                cwd=root_dir,
                capture_output=True,
                text=True,
                check=True
            )
            files = result.stdout.splitlines()
        else:
            files = glob.glob(os.path.join(root_dir, '/**'), recursive=True)

        for file_path in files:
            if '/test' in file_path or '_test.' in file_path:
                continue

            full_path = os.path.join(root_dir, file_path)
            # Determine extension (lowercased), files without an extension use empty string
            _, ext = os.path.splitext(file_path)
            ext = ext.lower().lstrip('.')  # Remove the leading dot

            # Skip files without valid extensions
            if ext not in VALID_EXTENSIONS:
                continue

            try:
                with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                    # Count lines
                    line_count = sum(1 for _ in f)
                ext_counts[ext] += line_count
            except (OSError, UnicodeError) as e:
                # Skip files we can't read
                print(f"Warning: could not read {full_path}: {e}", file=sys.stderr)

    except subprocess.CalledProcessError as e:
        print(f"Error running git ls-files: {e}", file=sys.stderr)
        sys.exit(1)

    return ext_counts


def main():
    args = parse_args()
    directory = args.directory

    if not os.path.isdir(directory):
        print(
            f"Error: '{directory}' is not a directory or does not exist.",
            file=sys.stderr,
        )
        sys.exit(1)

    counts = count_lines_by_extension(directory)

    # Print results sorted by descending line count
    print("Line counts by file extension:")
    for ext, total in sorted(counts.items(), key=lambda item: item[1], reverse=True):
        print(f"{total}\t{ext or '[no ext]'}")


if __name__ == "__main__":
    main()
