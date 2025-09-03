#curl -X 'POST' 'http://localhost:10068/v1/chat/completions'   -H 'accept: application/json'   -H 'Content-Type: application/json'   -d '{
#  "messages": [
#    {
#      "content": "Give me a clear and full explanation about the history of ThanksGiving.",
#      "role": "user"
#    }
#  ],
#  "model": "DeepSeek-Coder-V2-Instruct",
#  "stream": false
#}' &
#
#sleep 1

curl -X 'POST' 'http://localhost:10068/v1/chat/completions'   -H 'accept: application/json'   -H 'Content-Type: application/json'   -d '{
  "messages": [
    {
      "content": "Who are you?",
      "role": "user"
    }
  ],
  "model": "DeepSeek-Coder-V2-Instruct",
  "stream": false
}'  &

sleep 1

curl -X 'POST' 'http://localhost:10068/v1/chat/completions'   -H 'accept: application/json'   -H 'Content-Type: application/json'   -d '{
  "messages": [
    {
      "content": "Where is Beijing?",
      "role": "user"
    }
  ],
  "model": "DeepSeek-Coder-V2-Instruct",
  "stream": false
}'  &

sleep 1

curl -X 'POST' 'http://localhost:10068/v1/chat/completions'   -H 'accept: application/json'   -H 'Content-Type: application/json'   -d '{
  "messages": [
    {
      "content": "What you do?",
      "role": "user"
    }
  ],
  "model": "DeepSeek-Coder-V2-Instruct",
  "stream": false
}'  &

sleep 1

curl -X 'POST' 'http://localhost:10068/v1/chat/completions'   -H 'accept: application/json'   -H 'Content-Type: application/json'   -d '{
  "messages": [
    {
      "content": "Where is Beijing?",
      "role": "user"
    }
  ],
  "model": "DeepSeek-Coder-V2-Instruct",
  "stream": false
}'  &
