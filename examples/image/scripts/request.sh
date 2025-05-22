#!/bin/bash

# Make the POST request
# response=$(curl -i -s -X POST \
response=$(curl -s -X POST \
  -F file=@contract.png \
  http://127.0.0.1:8000/)

# Output the response
echo "$response"
