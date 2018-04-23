#!/usr/bin/env bundle exec ruby
require 'github_api'
require 'json'
require 'pry'

def usage
  puts "Usage: GITHUB_TOKEN=abc123... #{$0} GITHUB_ORGANIZATION REPO datestamp-to-start-from"
  exit 1
end

github_token = ENV['GITHUB_TOKEN']
usage unless github_token
org = ARGV.shift
usage unless org
repo = ARGV.shift
usage unless repo
since = ARGV.shift
usage unless since

GithubClient = Github.new(oauth_token: github_token)

# Given the name of a repo and a date in the past this finds all of the people
# who participated in PRs and associated them to the files that changed in the
# relevant commits.
def extract_reviews(org, repo, since)
  raise "Block required" unless block_given?

  pull_requests = GithubClient.pull_requests.all(org, repo, state: 'closed')

  while pull_requests.first.created_at > since
    pull_requests.each do |pull_request|
      author = pull_request['user']['login']
      comment_id = pull_request.issue_url.split('/').last

      comments = GithubClient.issues.comments.all(org, repo, number: comment_id).map do |comment|
        [
          comment['user']['login'],
          comment['body'],
        ]
      end

      files = GithubClient.pull_requests.files(org, repo, pull_request['number']).map {|r| r['filename'] }

      review = {
        pr: pull_request['number'],
        files: files,
        author: author,
        comments: comments,
      }
      yield review
    end

    pull_requests = pull_requests.next_page
  end
end

# This is a low-tech approach to streaming JSON to STDOUT
puts "["
is_first = true
extract_reviews(org, repo, since) do |review|
  puts "," unless is_first
  is_first = false
  puts review.to_json
  STDOUT.flush
end
puts "]"

