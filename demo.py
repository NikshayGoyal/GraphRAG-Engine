"""No-key demonstration of real retrieval on clearly synthetic evidence."""
import json
from src.demo_data import demo_index
from src.search import run_search

if __name__ == '__main__':
    index = demo_index()
    while True:
        question = input('Question (quit to exit): ').strip()
        if question.lower() in ('quit', 'exit'):
            break
        print(json.dumps(run_search(question, index, 'Graph + text'), indent=2))
