#!/bin/bash

# Check if correct number of arguments provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <model_url> <output_filename>"
    echo "Example: $0 'https://civitai.com/api/download/models/802821?type=Model&format=SafeTensor' 'my_model.safetensor'"
    exit 1
fi

# Load environment variables from .env file
if [ -f ~/.env ]; then
    export $(cat ~/.env | xargs)
fi

# Check if API key is available
if [ -z "$CIVIT_API_KEY" ]; then
    echo "Error: CIVIT_API_KEY not found in .env file"
    exit 1
fi

# Get arguments and convert URL to API format
INPUT_URL="$1"
# Expand ~ to full home directory path and make it absolute
OUTPUT_FILE=$(realpath -m "${2/#\~/$HOME}")

# Extract the model ID from the URL
MODEL_ID=$(echo "$INPUT_URL" | grep -o 'models/[0-9]*' | cut -d'/' -f2)
MODEL_URL="https://civitai.com/api/download/models/${MODEL_ID}"

# Create output directory if it doesn't exist
OUTPUT_DIR=$(dirname "$OUTPUT_FILE")
mkdir -p "$OUTPUT_DIR"

# Download the file
echo "Downloading model to $OUTPUT_FILE..."
curl -L -H "Authorization: Bearer $CIVIT_API_KEY" "$MODEL_URL" --output "$OUTPUT_FILE"

if [ $? -eq 0 ]; then
    echo "Download complete!"
else
    echo "Download failed!"
    exit 1
fi
