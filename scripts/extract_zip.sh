#!/bin/bash

# Define the source and destination directories
SOURCE_DIR="raw"
DEST_DIR="raw_extracted"

# Create the destination directory if it doesn't exist
mkdir -p "$DEST_DIR"

# Loop through each ZIP file in the source directory
for zip_file in "$SOURCE_DIR"/*.zip; do
    # Extract the ZIP file to the destination directory
    unzip -q "$zip_file" -d "$DEST_DIR"
done
