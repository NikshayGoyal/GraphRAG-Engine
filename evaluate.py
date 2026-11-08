"""Run the deterministic synthetic retrieval smoke benchmark without API access."""
import argparse
import json
from src.demo_data import demo_index, demo_questions
from src.evaluation import evaluate
from src.storage import atomic_json

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='evaluation_results.json')
    parser.add_argument('--top-k', type=int, default=3)
    args = parser.parse_args()
    result = evaluate(demo_index(), demo_questions(), args.top_k)
    atomic_json(args.output, result)
    print(json.dumps(result['summary'], indent=2))
