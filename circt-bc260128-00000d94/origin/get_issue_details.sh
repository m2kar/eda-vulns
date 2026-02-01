#!/bin/bash

REPO="llvm/circt"

# Get details for top 3 candidates
for issue_num in 8930 8332 8283; do
    echo ""
    echo "========================================"
    echo "ISSUE #$issue_num - Detailed View"
    echo "========================================"
    
    gh issue view $issue_num -R "$REPO" --json title,body,state,labels,url,number | jq .
done
