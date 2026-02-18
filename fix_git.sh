#!/bin/bash
cd /home/aswanth.cp@knackforge.com/Public/ML/Projects/VoiceAssistantwithLLM/chatapp-with-voice-and-openai-outline

# Abort any pending rebase
rm -rf .git/rebase-merge .git/rebase-apply 2>/dev/null

# Reset to origin/main
git reset --hard origin/main

# Apply all changes at once
git add .

# Commit with no secrets
git commit -m "Update codebase with optimizations and environment variable support"

echo "Ready to push. Run: git push --force"
