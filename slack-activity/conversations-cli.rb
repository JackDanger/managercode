require 'slack-ruby-client'
require 'fileutils'

Slack.configure do |config|
  config.token = ENV.fetch('SLACK_API_TOKEN')
end

class SlackCache

  DIR = ARGV[0] || './slack-cache'

  Epoch = Time.new(1970, 1, 1)

  def update
    puts "Caching messages in #{DIR}"
    @start_time = Time.now
    channels.each do |channel|
      update_channel(channel)
      break
    end
  end

  def update_channel(channel)
    cached_messages = retrieve_stored(channel)

    puts "Updating <#{channel['id']}|#{channel['name_normalized']}>"
    messages = update_messages(channel, cached_messages)

    store!(channel, messages.to_a)
  end

  def update_messages(channel, cached_messages)
    Enumerator.new do |enumerator|
      messages(channel['id'], cached_messages).sort_by { |m| m['ts'] }.each do |m| 
        if m['replies'] && !m['_replies_expanded']
          m['_replies_expanded'] = replies(channel['id'], m['ts']).map do |thread_m|
            thread_m['_username'] = user_map[thread_m['user']]
            thread_m['_parent_username'] = user_map[thread_m['parent_user_id']]
            thread_m['_ts'] = Epoch + thread_m['ts'].to_f
          end
        end
        if m['_ts'].nil?
          m['_username'] = user_map[m['user']]
          m['_parent_username'] = user_map[m['parent_user_id']]
          m['_ts'] = Epoch + m['ts'].to_f
        end
        enumerator.yield m
      end
    end
  end

  def messages(channel_id, cached_messages)
    # Unique the messages by timestamp
    cached_messages = cached_messages.each_with_object({}) {|m, acc| acc[m['ts']] = m }
    # and return the datastructure to it's original form, sorted by timestamp
    cached_messages = cached_messages.sort_by(&:first).map(&:last)

    Enumerator.new do |enumerator|
      sleep_factor = 1
      oldest = parse_oldest
      response = { 'has_more' => true }
      while response['has_more']
        begin

          puts "making a remote call for #{channel_id}, oldest: #{oldest} - #{Epoch + oldest.to_f} (oldest.object_id: #{oldest.object_id})"
          response = client.channels_history(channel: channel_id, oldest: oldest.to_f, limit: 1000, inclusive: true)
          puts "got #{response['messages'].size} remote messages"
          puts "have #{cached_messages.size} cached messages"
          require 'pry'
          binding.pry

          # When there's no remote data, just replay the cache and exit
          while response['messages'].empty? && cached_messages.any?
            print('-') && STDOUT.flush
            enumerator.yield cached_messages.shift
            break
          end

          oldest = response['messages'].first['ts'] if response['messages'].any?
          seen = Epoch.to_f

          response['messages'].reverse.each do |message|
            while cached_messages.any? && message['ts'] >= cached_messages.first['ts']
              # We've reached the start of the cached messages so let's process them
              cached_message = cached_messages.shift
              print("-") && STDOUT.flush
              enumerator.yield cached_message
              seen = cached_message['ts'].to_f
              oldest = cached_message['ts'] if oldest < cached_message['ts']
            end

            if seen < message['ts'].to_f
              print("+") && STDOUT.flush
              oldest = message['ts'] if oldest < message['ts']
              enumerator.yield message
              seen = message['ts'].to_f
            end
          end
          puts ''
          sleep_factor = 1
        rescue Slack::Web::Api::Errors::TooManyRequestsError
          sleep_amount = 10 + (2 ** sleep_factor)
          puts "Sleeping for #{sleep_amount}"
          sleep sleep_amount
          sleep_factor += 1
          next
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

  def stored?(channel)
    File.exist?(cache_file(channel))
  end

  def channels
    @channels ||= client.channels_list.channels.reject {|c| c['is_archived'] }
  end

  def store!(channel, data)
    puts ''
    puts "Storing <##{channel['id']}|#{channel['name_normalized']}>: #{data.size}"
    FileUtils.mkdir_p(DIR)
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
    File.join(DIR, channel['id'])
  end

  def parse_oldest
    @parse_oldest ||= Time.parse(ARGV[1]).to_f.to_s
  rescue
    @parse_oldest ||= Time.parse('2019-01-19 00:00:00 -0700').to_f.to_s
  end

  def client
    @client ||= Slack::Web::Client.new
  end

  class Channel
    attr_reader :id, :slack_cache

    def initialize(id, slack_cache)
      @id = id
      @slack_cache = slack_cache
    end
  end

  class Message
    attr_reader :id, :channel

    def initialize(id, channel)
      @id = id
      @channel = channel
    end
  end
end

SlackCache.new.update

require 'pry'
binding.pry

puts 'done'
