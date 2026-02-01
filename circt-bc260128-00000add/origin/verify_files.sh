#!/bin/bash

echo "=== File Integrity Verification ==="
echo ""

expected_files=(
    "source.sv"
    "error.txt"
    "status.json"
    "reproduce.log"
    "metadata.json"
    "root_cause.md"
    "analysis.json"
    "bug.sv"
    "error.log"
    "command.txt"
    "validation.json"
    "validation.md"
    "duplicates.json"
    "duplicates.md"
    "issue.md"
)

missing_files=()
empty_files=()
valid_files=()

for file in "${expected_files[@]}"; do
    if [ ! -f "$file" ]; then
        missing_files+=("$file")
    elif [ ! -s "$file" ]; then
        empty_files+=("$file")
    else
        valid_files+=("$file")
    fi
done

echo "✅ Valid files (${#valid_files[@]}):"
printf '  - %s\n' "${valid_files[@]}"

if [ ${#missing_files[@]} -gt 0 ]; then
    echo ""
    echo "❌ Missing files (${#missing_files[@]}):"
    printf '  - %s\n' "${missing_files[@]}"
fi

if [ ${#empty_files[@]} -gt 0 ]; then
    echo ""
    echo "⚠️  Empty files (${#empty_files[@]}):"
    printf '  - %s\n' "${empty_files[@]}"
fi

echo ""
if [ ${#missing_files[@]} -eq 0 ] && [ ${#empty_files[@]} -eq 0 ]; then
    echo "🎉 All files verified successfully!"
    exit 0
else
    echo "❌ File verification failed!"
    exit 1
fi
