require 'json'
require 'pry'

IGNORE_EMAILS = ENV['IGNORE_EMAILS'].to_s.split(',')

class Contributors
  attr_reader :dir_containing_repos, :since
  def initialize(dir_containing_repos, since)
    @dir_containing_repos = dir_containing_repos
    @since = since

    @contributors = {}
  end

  # Calculate the connections between repos by how many authors they share.
  def repo_connections_weighted_graph
    read_all_git_repos('repos')
    graph_links_between_git_objects
  end

  # Calculate the connections between files across all repos by how many authors they share.
  def file_connections_weighted_graph
    read_all_git_repos('files')
    graph_links_between_git_objects('files')
  end

  # Calculate the connections between committers into the same repositories.
  # Does not record any repository data, only the number of times two people
  # overlapped within any repository
  def contributor_connections_weighted_graph
    read_all_git_repos('files')

    all_commits = Hash.new { |h, k| h[k] = 0 }
    connection_counts = Hash.new { |h, k| h[k] = 0 }

    @contributors.each do |file, emails|
      emails.each do |email, count|
        all_commits[email] += count
      end
      # Do a quadratic calculation to find the matrix overlap of all
      # contributors to this project
      emails.keys.combination(2).each do |email_pair|
        key = email_pair_normalized(*email_pair)
        connection_counts[key] += [
          @contributors[file][email_pair.first],
          @contributors[file][email_pair.last],
        ].min
      end
    end

    graph_for(all_commits, connection_counts)
  end

  private

  def graph_links_between_git_objects(scope='repo')
    all_commits = Hash.new { |h, k| h[k] = 0 }
    connection_counts = Hash.new { |h, k| h[k] = 0 }
    @contributors.each do |obj, emails|
      emails.each do |email, count|
        all_commits[obj] += count
      end
    end

    # Do a quadratic calculation to find the number of shared contributors
    # this object (file or repo) has.
    @contributors.keys.combination(2).each do |obj_pair|
      @contributors[obj_pair.first].keys.each do |email|
        if @contributors[obj_pair.last][email]
          connection_counts[obj_pair] += [
            @contributors[obj_pair.first][email],
            @contributors[obj_pair.last][email],
          ].min
        end
      end
    end

    graph_for(all_commits, connection_counts)
  end

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
      } if count > 0
    end
    connection_counts.each do |pair, count|
      graph[:links] << {
        source: pair.first,
        target: pair.last,
        value: count,
      } if count > 0
    end
    graph
  end

  def read_all_git_repos(scope='repo')
    return if @processed

    # If `scope` == 'file' the cmd will generate output in the following
    # format:
    #
    #    BREAK: github@jackcanty.com
    #    
    #    .ruby-version
    #    Gemfile
    #    Gemfile.lock
    #    pr_commenters.rb
    #    BREAK: github@jackcanty.com
    #    
    #    drive/Gemfile
    #    drive/Gemfile.lock
    #    drive/app.rb

    Dir[dir_containing_repos + '/*/.git'].each do |git_dir|
      dir = File.dirname(git_dir)
      basename = File.basename(dir)
      email = nil
      Dir.chdir(dir) do
        warn dir
        if scope == 'files'
          cmd = "git log --name-only --format='BREAK: %ae' --since='#{since}'"
          %x|#{cmd}|.each_line do |line|
            if line =~ /^BREAK: /
              email = line.split('BREAK: ').last.chomp
            elsif line =~ /^\s$/
              # skip
            else
              next if IGNORE_EMAILS.include?(email)
              full_path = File.join(basename, line.chomp)
              ## Uncomment to show verbose logging
              # warn "#{email} -> #{full_path}"
              @contributors[full_path] ||= {}
              @contributors[full_path][email] ||= 0
              @contributors[full_path][email] += 1
            end
          end
        else
          cmd = "git log --format='%ae' --since='#{since}'"
          %x|#{cmd}|.each_line do |line|
            email = line.chomp
            next if IGNORE_EMAILS.include?(email)
            ## Uncomment to show verbose logging
            # warn "#{email} -> #{basename}"
            @contributors[basename] ||= {}
            @contributors[basename][email] ||= 0
            @contributors[basename][email] += 1
          end
        end
      end
    end
    @processed = true
  end

  def group_from_email_domain(email)
    @email_domains ||= {}
    parts = email.split('@')
    # The incalculable case:
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
  o.on '-t', '--type TYPE', String, "Either 'contributors', 'repos', or 'files'" do |value|
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
  puts contributors.contributor_connections_weighted_graph.to_json
when 'repos'
  puts contributors.repo_connections_weighted_graph.to_json
when 'files'
  puts contributors.file_connections_weighted_graph.to_json
else
  raise "Unknown calculation type: #{options[:type]}. Valid options are 'contributors' and 'files'"
end

