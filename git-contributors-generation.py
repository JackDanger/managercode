import subprocess
import os
import json
from collections import defaultdict
from pathlib import Path


def get_github_team_members(org, team):
    result = subprocess.run(
        ["gh", "team", "list", "--org", org, "--json", "members", "--role", team],
        capture_output=True,
        text=True,
        check=True,
    )
    members_data = json.loads(result.stdout)
    return [member['login'] for member in members_data['members']]


def get_git_history(directory):
    git_log_cmd = ["git", "-C", directory, "log", "--name-only", "--pretty=format:%an"]
    result = subprocess.run(git_log_cmd, capture_output=True, text=True, check=True)
    history = result.stdout.split('\n')

    user_files = defaultdict(set)
    current_user = None

    for line in history:
        if line.strip() == '':
            current_user = None
        elif not line.startswith(' '):
            current_user = line.strip()
        elif current_user:
            user_files[current_user].add(line.strip())

    return user_files


def build_connection_graph(user_files):
    connections = defaultdict(lambda: defaultdict(int))

    users = list(user_files.keys())
    for i in range(len(users)):
        for j in range(i + 1, len(users)):
            user_a, user_b = users[i], users[j]
            common_files = user_files[user_a] & user_files[user_b]
            if common_files:
                connections[user_a][user_b] += len(common_files)
                connections[user_b][user_a] += len(common_files)

    return connections


def main():
    org = sys.argv[1]
    repo_dir = sys.argv[2]
    team = None
    if len(sys.argv) > 3:
        team = sys.argv[3]


    # Get GitHub team members
    if team is not None:
        team_members = get_github_team_members(org, team)

    # Get git history of all repos in the directory
    user_files = defaultdict(set)
    for repo in Path(repo_dir).iterdir():
        if repo.is_dir() and (repo / ".git").exists():
            user_files = get_git_history(str(repo))
            for user, files in user_files.items():
                user_files[user].update(files)

    if team is not None:
        # Filter only team members
        user_files = {user: files for user, files in user_files.items() if user in team_members}

    # Build connection graph
    connection_graph = build_connection_graph(user_files)

    # Print JSON file
    json.dump(connection_graph, sys.stdout, indent=2)


if __name__ == "__main__":
    main()
