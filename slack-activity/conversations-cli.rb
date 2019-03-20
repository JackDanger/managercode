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
    end
  end

  def update_channel(channel)
    retrieve_stored(channel)
      puts "already stored #{channel['name']}"
      return
    end
    store!(channel, formatted_messages(channel).to_json)
  end

  def formatted_messages(channel)
    Enumerator.new do
      messages(channel['id']).each do |m| 
        message = m.slice('user', 'text', 'ts', 'parent_user_id', 'client_msg_id', 'reactions')
        if m['replies']
          replies = replies(channel['id'], m['ts'])
          message = { thread: replies['messages'] }
          message[:thread].each do |mm|
            mm['_username'] = user_map[mm['user']]
            mm['_parent_username'] = user_map[mm['parent_user_id']]
            mm['_ts'] = Epoch + mm['ts'].to_f
          end
        else
          message['_username'] = user_map[message['user']]
          message['_parent_username'] = user_map[message['parent_user_id']]
          message['_ts'] = Epoch + message['ts'].to_f
        end
        yield message
      end
    end
  end

  def messages(channel_id)
    Enumerator.new do
      oldest = parse_oldest
      response = { 'has_more' => true }
      while response['has_more']
        begin
        response = client.channels_history(channel: channel_id, oldest: oldest, limit: 1000)
        break if response['messages'].empty?

        oldest = response['messages'].first['ts']
        response['messages'].each { |m| yield m }
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
    response = client.channels_replies(channel: channel['id'], thread_ts: m['ts'])
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
    puts "Storing <##{channel['id']}|#{channel['name_normalized']}>: #{data.size}"
    FileUtils.mkdir_p(DIR)
    File.open(cache_file(channel), 'w') do |f|
      f.write data
    end
  end

  def cache_file(channel)
    File.join(DIR, channel['id'])
  end

  def parse_oldest
    @parse_oldest ||= Time.parse(ARGV[1])
  rescue
    @parse_oldest ||= Time.parse('2019-01-19 00:00:00 -0700')
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
