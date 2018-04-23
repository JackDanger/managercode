require 'googleauth'
require 'google/apis/drive_v3'

OOB_URI = 'urn:ietf:wg:oauth:2.0:oob'

# Get the environment configured authorization
unless ENV['CLIENT_SECRET']
  puts "Usage: CLIENT_SECRET=/path/to/secret.json ruby #{$0}"
  exit 1
end
scope = 'https://www.googleapis.com/auth/drive'
authorizer = Google::Auth::ServiceAccountCredentials.make_creds(
  json_key_io: File.open(ENV['CLIENT_SECRET']),
  scope: scope)
authorization = authorizer.fetch_access_token!

service = Google::Apis::DriveV3::DriveService.new
service.client_options.application_name = "Jack Danger's Managercode"
service.authorization = authorization
response = service.list_files(page_size: 10, fields: 'nextPageToken, files(id, name)')

require 'pry'
binding.pry

puts "what"
