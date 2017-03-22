import os
import csv

def get_accuracy(file_name):
    with open(os.path.join('logs', file_name), newline='') as f:
        reader = csv.reader(f)
        last = iter(reader)
        for row in reader:
            last = row
        return float(last[2])

if __name__ == '__main__':
    results = {}
    for file_name in os.listdir('logs'):
        param_set = tuple(map(float, file_name.split('.csv')[0].split('-')))
        results[param_set] = get_accuracy(file_name)
    rev_reults = {v: k for k, v in results.items()}
    top_scores = sorted(rev_reults.keys(), reverse=True)
    for score in top_scores[:10]:
        print(score, rev_reults[score])