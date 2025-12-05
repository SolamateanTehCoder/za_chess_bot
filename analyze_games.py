import json
from collections import defaultdict

# Load all games
games = []
with open('games.jsonl', 'r') as f:
    for line in f:
        games.append(json.loads(line))

print(f'Total games: {len(games)}\n')

# First 10 games
first10_results = defaultdict(int)
for game in games[:10]:
    first10_results[game['result']] += 1

# Last 10 games
last10_results = defaultdict(int)
for game in games[-10:]:
    last10_results[game['result']] += 1

# All games
all_results = defaultdict(int)
all_rewards = []
for game in games:
    all_results[game['result']] += 1
    avg_reward = sum(m['reward'] for m in game['moves']) / len(game['moves']) if game['moves'] else 0
    all_rewards.append(avg_reward)

print('First 10 games:')
print(f'  Wins: {first10_results.get("Win", 0)}, Losses: {first10_results.get("Loss", 0)}, Draws: {first10_results.get("Draw", 0)}')

print('\nLast 10 games:')
print(f'  Wins: {last10_results.get("Win", 0)}, Losses: {last10_results.get("Loss", 0)}, Draws: {last10_results.get("Draw", 0)}')

print(f'\nOverall ({len(games)} games):')
total = len(games)
win_pct = (all_results['Win'] / total) * 100
loss_pct = (all_results['Loss'] / total) * 100
draw_pct = (all_results['Draw'] / total) * 100
print(f'  Wins: {all_results["Win"]} ({win_pct:.1f}%)')
print(f'  Losses: {all_results["Loss"]} ({loss_pct:.1f}%)')
print(f'  Draws: {all_results["Draw"]} ({draw_pct:.1f}%)')

print(f'\nAverage reward per game: {sum(all_rewards) / len(all_rewards):.4f}')
print(f'Average move time (ms): {sum(m["move_time_ms"] for g in games for m in g["moves"]) / sum(len(g["moves"]) for g in games):.2f}')

# Improvement metrics
print('\n' + '='*60)
print('IMPROVEMENT ANALYSIS')
print('='*60)

first_100_results = defaultdict(int)
for game in games[:100]:
    first_100_results[game['result']] += 1

last_100_results = defaultdict(int)
for game in games[-100:]:
    last_100_results[game['result']] += 1

first_100_win_pct = (first_100_results['Win'] / 100) * 100
last_100_win_pct = (last_100_results['Win'] / 100) * 100

print(f'\nFirst 100 games:')
print(f'  Win rate: {first_100_win_pct:.1f}%')
print(f'  Wins: {first_100_results["Win"]}, Losses: {first_100_results["Loss"]}, Draws: {first_100_results["Draw"]}')

print(f'\nLast 100 games:')
print(f'  Win rate: {last_100_win_pct:.1f}%')
print(f'  Wins: {last_100_results["Win"]}, Losses: {last_100_results["Loss"]}, Draws: {last_100_results["Draw"]}')

improvement = last_100_win_pct - first_100_win_pct
print(f'\nWin rate improvement: {improvement:+.1f}%')
