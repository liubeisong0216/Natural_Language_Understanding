# Add all your Python calls here:
python truthfulqa.py facebook/opt-1.3b --no-demos
python truthfulqa.py facebook/opt-1.3b --demos demonstrations.txt
python truthfulqa.py facebook/opt-1.3b --system-prompt "Actually," --no-demos
python truthfulqa.py facebook/opt-1.3b --demos demonstrations.txt --system-prompt "Actually,"
# ... etc.