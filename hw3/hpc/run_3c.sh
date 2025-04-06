# Add all your Python calls here:
python truthfulqa.py facebook/opt-1.3b --demos demonstrations.txt --system-prompt "Think carefully and despite the popular belief,"
python truthfulqa.py facebook/opt-1.3b --demos demonstrations.txt --system-prompt "Challenge common assumptions and verify the facts before responding,"
python truthfulqa.py facebook/opt-1.3b --demos demonstrations.txt --system-prompt "Analyze reasoning step by step before giving the final answer,"
# ... etc.