require 'slop'
require 'json'
require 'pry'

class Contributors
  attr_reader :dir_containing_repos, :since
  def initialize(dir_containing_repos, since)
    @dir_containing_repos = dir_containing_repos
    @since = since

    @contributors = Hash.new { |h1, k1| h1[k1] = Hash.new { |h2, k2| h2[k2] = 0 } }
  end

  # Calculate the connections between repos by how many authors they share.
  def repo_weighted_graph
    read_all_git_repos
    all_commits = Hash.new { |h, k| h[k] = 0 }
    connection_counts = Hash.new { |h, k| h[k] = 0 }
    @contributors.each do |project, emails|
      emails.each do |email, count|
        all_commits[project] += count
      end
    end
    # Do a quadratic calculation to find the number of shared contributors
    # this repo has.
    @contributors.keys.combination(2).each do |project_pair|
      @contributors[project_pair.first].keys.each do |email|
        if @contributors[project_pair.last][email]
          connection_counts[project_pair] += [
            @contributors[project_pair.first][email],
            @contributors[project_pair.last][email],
          ].min
        end
      end
    end

    graph_for(all_commits, connection_counts)
  end

  # Calculate the connections between committers into the same repositories.
  # Does not record any repository data, only the number of times two people
  # overlapped within any repository
  def contributor_weighted_graph
    read_all_git_repos

    all_commits = Hash.new { |h, k| h[k] = 0 }
    connection_counts = Hash.new { |h, k| h[k] = 0 }

    @contributors.each do |project, emails|
      emails.each do |email, count|
        all_commits[email] += count
      end
      # Do a quadratic calculation to find the matrix overlap of all
      # contributors to this project
      emails.keys.combination(2).each do |email_pair|
        key = email_pair_normalized(*email_pair)
        connection_counts[key] += [
          @contributors[project][email_pair.first],
          @contributors[project][email_pair.last],
        ].min
      end
    end

    graph_for(all_commits, connection_counts)
  end

  private

  def graph_for(all_commits, connection_counts)
    graph = {
      nodes: [],
      links: [],
    }

    all_commits.each do |key, count|
      graph[:nodes] << {
        id: key,
        group: group_from_email_domain(key),
        count: count,
      }
    end
    connection_counts.each do |pair, count|
      graph[:links] << {
        source: pair.first,
        target: pair.last,
        value: count,
      }
    end
    graph
  end

  def read_all_git_repos
    return if @processed
    cmd = "git log --format='%ae' --since='#{since}'"
    Dir[dir_containing_repos + '/*/.git'].each do |git_dir|
      dir = File.dirname(git_dir)
      basename = File.basename(dir)
      Dir.chdir(dir) do
        %x|#{cmd}|.each_line do |sha|
          add_contribution(basename, sha.chomp)
        end
      end
    end
    @processed = true
  end

  def add_contribution(project, email)
    parts = project.split('/')
    project_name = parts.last

    @contributors[project_name][email] += 1
  end

  def group_from_email_domain(email)
    @email_domains ||= {}
    parts = email.split('@')
    # Incalculable case
    return 1 if parts.size < 2
    domain = parts.last

    @email_domains[domain] ||= @email_domains.length + 1

    @email_domains[domain]
  end

  def email_pair_normalized(email1, email2)
    if email1 < email2
      [email1, email2]
    else
      [email2, email1]
    end
  end
end

require 'optparse'
options = {}
option_parser = OptionParser.new do |o|
  o.banner = "USAGE: #{$0} -d /path/to/repos -s '1 day ago' --type TYPE"
  o.on '-d', '--repo_dir DIR', String, "A directory containing git repos you wish to analyze" do |value|
    options[:dir_containing_repos] = value
  end
  o.on '-s', '--since TIME', String, "an argument to git's --since argument (e.g. '1 week ago')" do |value|
    options[:since] = value
  end
  o.on '-t', '--type TYPE', String, "Either 'contributors' or 'repos'" do |value|
    options[:type] = value
  end
  o.on '-h', '--help' do
    puts o.banner
    exit 1
  end
end

option_parser.parse!

unless options[:dir_containing_repos] && options[:since] && options[:type]
  puts option_parser.banner
  exit 1
end

contributors = Contributors.new(options[:dir_containing_repos], options[:since])

case options[:type]
when 'contributors'
  puts contributors.contributor_weighted_graph.to_json
when 'repos'
  puts contributors.repo_weighted_graph.to_json
else
  raise "Unknown calculation type: #{options[:type]}. Valid options are 'contributors' and 'repos'"
end

