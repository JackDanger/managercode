#!/usr/bin/env python3
import argparse
import http.server
import re
import os
import shutil
import socketserver
import subprocess
import sys
import json
from collections import defaultdict
from pathlib import Path


def get_github_team_members(org, team):
    members_url_cmd = f"gh api /orgs/{org}/teams/{team}"
    output = subprocess.check_output(members_url_cmd, shell=True).decode('utf-8')
    team_data = json.loads(output)
    members_url = team_data['members_url']
    members_path = members_url.replace('https://api.github.com').split('{')[0]


    result = subprocess.run(
        ["gh", "api", members_path],
        capture_output=True,
        text=True,
        check=True,
    )
    members_data = json.loads(result.stdout)
    return [member['login'] for member in members_data['members']]


def get_git_history(directory, user_names, ignore, since):
    user_files = defaultdict(set)

    git_log_cmd = ["git", "-C", directory, "log", "--name-only", "--format=BREAK: %cl %an", f"--since={since}"]
    result = subprocess.run(git_log_cmd, capture_output=True, text=True, check=True)
    history = result.stdout.split('\n')

    current_author_email = None

    breakline = re.compile('^BREAK: ([^ ]+) (.*)')

    for line in history:
        match = breakline.match(line)

        if match:
            current_author_email = match.group(1)
            user_names[current_author_email] = match.group(2)
        elif line.strip() == '':
            pass
        elif current_author_email:
            user_files[current_author_email].add(line.strip())

    return user_files


def build_connection_graph(user_files):
    connections = []

    users = list(user_files.keys())
    for i in range(len(users)):
        for j in range(i + 1, len(users)):
            user_a, user_b = users[i], users[j]
            common_files = user_files[user_a] & user_files[user_b]
            if common_files:
                connections.append({
                    'source': user_a,
                    'target': user_b,
                    'value': len(common_files),
                })

    return connections


def main():
    parser = argparse.ArgumentParser(description='Generate git contributor graph')

    parser.add_argument('-d', '--directory', required=True, help='parent directory of all git repos')
    parser.add_argument('-o', '--organization', help='GitHub organization name')
    parser.add_argument('-t', '--team', help='GitHub team name')
    parser.add_argument('-i', '--ignore', nargs='+', default=[], help='List of usernames to ignore (e.g. "-i ansible root")')
    parser.add_argument('-s', '--since', default='90 days ago', help='how far back to analyze')

    args = parser.parse_args()

    directory = args.directory
    organization = args.organization
    team = args.team
    ignore = args.ignore
    since_date = args.since
    since_slug = since_date.replace(' ', '-')

    output_dir = f'git.{since_slug}'

    # Get GitHub team members
    if team is not None:
        team_members = get_github_team_members(organization, team)

    # Make a mapping of commit emails to names in case we want to show those during rendering
    user_names = dict()
    # Get git history of all repos in the directory
    user_files = defaultdict(set)
    for repo in Path(directory).iterdir():
        if repo.is_dir() and (repo / ".git").exists():
            history = get_git_history(str(repo), user_names, ignore=ignore, since='90 days ago')
            for user, files in history.items():
                for filename in files:
                    file = f'{repo}/{filename}'
                    user_files[user].add(file)

    if team is not None:
        # Filter only team members
        user_files = {user: files for user, files in user_files.items() if user in team_members}


    # Remove ignored users
    for ignored in ignore:
        if ignored in user_files:
            del user_files[ignored]
        if ignored in user_names:
            del user_names[ignored]

    # Build connection graph
    connection_graph = build_connection_graph(user_files)
    data = {
        'nodes': [ { 'id': email, 'group': 1, 'count': len(user_files[email])} for email in user_files],
        'links': connection_graph,
    }

    # Create a directory for the rendered graph
    os.makedirs(output_dir, exist_ok=True)

    # Copy HTML into the rendered directory
    shutil.copy('index.html', output_dir)

    output_file = f"{output_dir}/graphData.json"
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)

    # Start the web server
    os.chdir(output_dir)
    PORT = 5050

    with socketserver.TCPServer(("", PORT), http.server.SimpleHTTPRequestHandler) as httpd:

        print(f"Serving at port {PORT}")
        subprocess.run(['open', f'http://localhost:{PORT}/'], capture_output=False, text=True, check=False)
        httpd.serve_forever()

  
if __name__ == "__main__":
    main()
