#!/bin/bash

response=$(curl -s -X 'POST' \
  'http://127.0.0.1:8000/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "content": "2+2"
}')

echo "$response" | jq
