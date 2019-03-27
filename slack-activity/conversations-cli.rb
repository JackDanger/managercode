require 'slack-ruby-client'
require 'yaml'
require 'fileutils'

Slack.configure do |config|
  config.token = ENV.fetch('SLACK_API_TOKEN')
end

Epoch = Time.new(1970, 1, 1, 0, 0, 1, 0)


class SlackCache

  attr_reader :manifest

  SYNC_BATCH_SIZE = 1_000
  SYNC_REST_DELAY_IN_SECONDS = 60 * 60 * 4 # 4 hours

  def initialize
    @manifest = Manifest.new(self)
  end

  def update
    puts "Caching messages in #{dir}"
    @start_time = Time.now
    channels.each do |channel|
      if manifest.recently_synced?(channel['id'])
        puts "Skipping <#{channel['id']}|#{channel['name_normalized']} – synced #{manifest.last_synced(channel['id'])} seconds ago"
        next
      end
      update_channel(channel)
    end
    puts "finished in #{Time.now - @start_time}"
  end

  def most_recent_ts_in(messages)
    messages.max {|a, b| a['ts'] <=> b['ts'] }
  end

  def update_channel(channel)
    puts "Updating <#{channel['id']}|#{channel['name_normalized']}>"

    enqueued_to_store = []
    messages(channel['id']).each do |message|
      enqueued_to_store << format_message(message, channel['id'])
      if enqueued_to_store.size > SYNC_BATCH_SIZE
        store!(channel, enqueued_to_store)
        enqueued_to_store.clear
      end
    end
    store!(channel, enqueued_to_store) if enqueued_to_store.any?
  end

  def format_message(m, channel_id)
    if m['replies'] && !m['_replies_expanded']
      m['_replies_expanded'] = replies(channel_id, m['ts']).map do |thread_m|
        thread_m['_username'] = user_map[thread_m['user']]
        thread_m['_parent_username'] = user_map[thread_m['parent_user_id']]
        thread_m['_ts'] = Epoch + thread_m['ts'].to_f
        thread_m
      end
    end
    if m['reactions']
      m['reactions'].each do |reaction|
        reaction['users'] = reaction['users'].map {|uid| "<##{uid}|@#{user_map[uid]}>" }
      end
    end
    if m['_ts']
      timestamp = Time.parse(m['_ts'])
    else
      m['_username'] = user_map[m['user']]
      m['_parent_username'] = user_map[m['parent_user_id']]
      timestamp = Epoch + m['ts'].to_f
      m['_ts'] = timestamp.to_s
    end
    m
  end

  def messages(channel_id)
    # Start at either the beginning of the channel history or where we left off.
    oldest = manifest.last_synced_ts(channel_id) || Epoch.to_f.to_s

    Enumerator.new do |enumerator|
      api_options = { channel: channel_id, oldest: oldest }
      # puts "Getting all #{channel_id} history from #{oldest.inspect} to #{parse_latest.inspect}"
      client.conversations_history(api_options) do |response|
        # Sort the messages by increasing values - oldest to newest, just as
        # they're displayed in the UI
        messages = response['messages'].sort_by {|m| m['ts'] }
        manifest.no_results!(channel_id) if messages.empty?
        messages.each do |message|
          print("+") && STDOUT.flush
          enumerator.yield message
        end
      end
    end
  end

  def replies(channel_id, thread_ts)
    print('T') && STDOUT.flush
    # puts "making remote call for thread: #{channel_id}/#{thread_ts}"
    response = client.channels_replies(channel: channel_id, thread_ts: thread_ts)
    response['messages']
  rescue Slack::Web::Api::Errors::TooManyRequestsError
    sleep 20
    retry
  end

  def user_map
    @user_map ||= client.users_list['members'].reduce({}) do |acc, user|
      acc.update user['id'] => user['name']
    end
  end

  def channels
    @channels ||= client.channels_list.channels
  end

  def dir
    @dir ||= File.join(ARGV[0] || './slack-cache')
  end

  def store!(channel, messages)
    FileUtils.mkdir_p(dir)

    cached_messages = retrieve_stored(channel)
    data = (cached_messages + messages).sort_by { |m| m['ts'] }
    puts ''
    puts "Storing <##{channel['id']}|#{channel['name_normalized']}>: #{messages.size}/#{data.size}"

    manifest.set_last_synced_ts!(channel['id'], data.last['ts'])

    File.open(cache_file(channel), 'w') do |f|
      f.write data.to_json
    end
  end

  def retrieve_stored(channel)
    JSON.parse(File.read(cache_file(channel)))
  rescue JSON::ParserError, Errno::ENOENT
    []
  end

  def cache_file(channel)
    File.join(dir, "#{channel['name_normalized']}-#{channel['id']}")
  end

  def client
    @client ||= Slack::Web::Client.new
  end

  class Manifest
    def initialize(slack_cache)
      @slack_cache = slack_cache
    end

    def data
      @data ||= read!
    end

    def recently_synced?(channel_id)
      return unless last_synced(channel_id)
      (Time.now.to_f - last_synced(channel_id).to_f) < SYNC_REST_DELAY_IN_SECONDS
    end

    def last_synced(channel_id)
      data[channel_id] ||= {}
      data[channel_id]['performed_at']
    end

    def no_results!(channel_id)
      data[channel_id] ||= {}
      data[channel_id]['performed_at'] = Time.now
      write!
    end

    def set_last_synced_ts!(channel_id, ts)
      data[channel_id] ||= {}
      data[channel_id]['latest'] = ts
      data[channel_id]['performed_at'] = Time.now
      write!
    end

    def last_synced_ts(channel_id)
      data[channel_id] ||= {}
      data[channel_id]['latest']
    end

    private

    def write!
      open(filename, 'w') { |f| f.write data.to_yaml }
    end

    def read!
      if File.exist?(filename)
        YAML.load_file(filename)
      else
        {}
      end
    end

    def filename
      @filename ||= File.join(@slack_cache.dir, 'manifest.yml')
    end
  end

end

SlackCache.new.update

puts 'done'
