require 'slack-ruby-client'
require 'fileutils'

Slack.configure do |config|
  config.token = ENV['SLACK_API_TOKEN']
end

client = Slack::Web::Client.new
channels = client.channels_list.channels.reject {|c| c['is_archived'] }

CACHE_DIR = ARGV[0] || '/tmp/slack-cache'
puts "Caching messages in #{CACHE_DIR}"
OLDEST = Time.parse('2018-01-01 00:00:00 -0700')
START_TIME = Time.now

def cache_file(channel)
  File.join(CACHE_DIR, channel['id'])
end

def stored?(channel)
  File.exist?(cache_file(channel))
end

def store!(channel, data)
  puts "Storing <##{channel['id']}|#{channel['name_normalized']}>: #{data.size}"
  FileUtils.mkdir_p(CACHE_DIR)
  File.open(cache_file(channel), 'w') do |f|
    f.write data
  end
end

sleep_factor = 1

channels.each do |channel|
  next if stored?(channel)
  messages = []
  response = { 'has_more' => true }
  oldest = OLDEST.to_i

  while response['has_more']
    begin
    response = client.channels_history(channel: channel['id'], oldest: oldest, limit: 1000)
    next if response['messages'].empty?
    sleep_factor = 1
    rescue Slack::Web::Api::Errors::TooManyRequestsError
      sleep_amount = 5 + (2 ** sleep_factor)
      puts "Sleeping for #{sleep_amount}"
      sleep sleep_amount
      sleep_factor += 1
      next
    end
    oldest = response['messages'].first['ts']
    response['messages'].each do |m| 
      messages << m.slice('user', 'text', 'parent_user_id', 'client_msg_id', 'replies', 'reactions')
    end
  end
  store!(channel, messages.to_json)
end

require 'pry'
binding.pry

puts 'done'

# Download the last 1 week of messages from all channels
