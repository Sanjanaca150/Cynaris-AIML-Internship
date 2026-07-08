
name = 'Cynaris AI Intern'
batch = 2026
accuracy = 0.947
print(f'Welcome, {name} | Batch: {batch} | Model Accuracy: {accuracy:.1%}')


scores = [85, 92, 78, 95, 88, 76, 91]
print(f'All scores:  {scores}')
print(f'Top 3:       {sorted(scores, reverse=True)[:3]}')
print(f'Average:     {sum(scores)/len(scores):.2f}')


student = {'name': 'Priya', 'week': 1, 'completed': True}
for key, value in student.items():
    print(f'  {key}: {value}')


def normalize(data):
    lo, hi = min(data), max(data)
    return [(x - lo) / (hi - lo) for x in data]

print('Normalized scores:', [round(v, 2) for v in normalize(scores)])
