#!/usr/bin/env python3
import argparse
import boto3
import http.server
import mimetypes
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


def get_git_commit_counts(directory, user_commits, ignore, since):

    cmd = f"git -C {directory} log --format='%ae' --since '{since}' | sort | uniq -c"
    counts = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    for line in counts.stdout.split("\n"):
        columns = re.split(r'\s+', line.strip(), maxsplit=1)
        if len(columns) > 1:
            count, email = columns
            username = email.split('@')[0]
            user_commits[username] += int(count)

    return user_commits

def get_git_files_changed(directory, user_names, ignore, since):
    user_files = defaultdict(set)

    git_log_cmd = ["git", "-C", directory, "log", "--name-only", "--format=BREAK: %cl %an", f"--since={since}"]
    result = subprocess.run(git_log_cmd, capture_output=True, text=True, check=True)
    history = result.stdout.split('\n')

    current_username = None

    breakline = re.compile('^BREAK: ([^ ]+) (.*)')

    for line in history:
        match = breakline.match(line)

        if match:
            current_username = match.group(1)
            user_names[current_username] = match.group(2)
        elif line.strip() == '':
            pass
        elif current_username:
            user_files[current_username].add(line.strip())

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


def upload_directory_to_s3(src_dir, publish):
    # Extract bucket name and path from the publish parameter
    if publish.startswith('s3://'):
        publish = publish[5:]
    bucket_name, s3_path = publish.split('/', 1)
    
    s3_client = boto3.client('s3')
    bucket_location = s3_client.get_bucket_location(Bucket=bucket_name)
    public_url = f"https://s3.{bucket_location['LocationConstraint']}.amazonaws.com/{bucket_name}/{s3_path}"

    print(f"aws s3 cp --recursive --acl public-read {src_dir} {publish}")
    
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            local_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_path, src_dir)
            s3_file_path = os.path.join(s3_path, relative_path).replace("\\", "/")
            
            # Determine the content type
            content_type, _ = mimetypes.guess_type(local_path)
            if not content_type:
                if file.endswith('.html'):
                    content_type = 'text/html'
                elif file.endswith('.json'):
                    content_type = 'application/json'
                else:
                    content_type = 'binary/octet-stream'

            s3_client.upload_file(
                    local_path,
                    bucket_name,
                    s3_file_path,
                    ExtraArgs={'ACL': 'public-read', 'ContentType': content_type})
    return public_url


def main():
    parser = argparse.ArgumentParser(description='Generate git contributor graph')

    parser.add_argument('-d', '--directory', required=True, help='parent directory of all git repos')
    parser.add_argument('-o', '--organization', help='GitHub organization name')
    parser.add_argument('-t', '--team', help='GitHub team name')
    parser.add_argument('-i', '--ignore', nargs='+', default=[], help='List of usernames to ignore (e.g. "-i ansible root")')
    parser.add_argument('-s', '--since', default='90 days ago', help='how far back to analyze')
    parser.add_argument('-p', '--publish', help='an S3 bucket in which to publish the results')

    args = parser.parse_args()

    directory = args.directory
    organization = args.organization
    team = args.team
    ignore = args.ignore
    since_date = args.since
    publish = args.publish

    since_slug = since_date.replace(' ', '-')
    output_dir = f'git.{since_slug}'

    # Get GitHub team members
    if team is not None:
        team_members = get_github_team_members(organization, team)

    # Make a mapping of commit emails to names in case we want to show those during rendering
    user_names = dict()
    # Get git history of all repos in the directory
    user_files = defaultdict(set)
    user_commits = defaultdict(int)
    for repo in Path(directory).iterdir():
        if repo.is_dir() and (repo / ".git").exists():

            files_changed = get_git_files_changed(str(repo), user_names, ignore=ignore, since=since_date)
            for user, files in files_changed.items():
                for filename in files:
                    file = f'{repo}/{filename}'
                    user_files[user].add(file)

            get_git_commit_counts(str(repo), user_commits, ignore=ignore, since=since_date)


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
        'nodes': [ { 'id': username, 'group': 1, 'count': user_commits[username]} for username in user_files],
        'links': connection_graph,
    }

    # Create a directory for the rendered graph
    os.makedirs(output_dir, exist_ok=True)

    # Copy HTML into the rendered directory
    shutil.copy('index.html', output_dir)

    output_file = f"{output_dir}/graphData.json"
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)

    if publish:
        http_dir_guess = upload_directory_to_s3(output_dir, publish)
        subprocess.run(['open', http_dir_guess + '/index.html'], capture_output=False, text=True, check=False)
    else:

        # Start the web server
        os.chdir(output_dir)
        PORT = 5050

        with socketserver.TCPServer(("", PORT), http.server.SimpleHTTPRequestHandler) as httpd:

            print(f"Serving at port {PORT}")
            subprocess.run(['open', f'http://localhost:{PORT}/'], capture_output=False, text=True, check=False)
            httpd.serve_forever()

  
if __name__ == "__main__":
    main()
