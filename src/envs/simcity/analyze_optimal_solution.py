"""
Analysis Script for SimCity Optimal Solution

This script analyzes the optimal solution found by the MILP solver
and provides insights for comparison with RL training results.
"""

import json
import numpy as np


def load_solution(filename: str = 'simcity_optimal_solution.json'):
    """Load the optimal solution from JSON file."""
    with open(filename, 'r') as f:
        return json.load(f)


def analyze_building_strategy(solution):
    """Analyze the building placement strategy."""
    print("=" * 60)
    print("BUILDING STRATEGY ANALYSIS")
    print("=" * 60)
    
    building_counts = {'Park': 0, 'House': 0, 'Shop': 0}
    player_buildings = {'P1': 0, 'P2': 0, 'P3': 0}
    
    for action in solution['building_sequence']:
        if action['building'] != 'No-op':
            building_counts[action['building']] += 1
            player_buildings[action['player']] += 1
    
    print(f"Total Buildings Placed: {sum(building_counts.values())}")
    print("\nBuilding Distribution:")
    for building, count in building_counts.items():
        print(f"  {building}: {count}")
    
    print("\nPlayer Activity:")
    for player, count in player_buildings.items():
        print(f"  {player}: {count} buildings")
    
    # Analyze timing - when do players stop building?
    last_building_turn = {}
    for action in solution['building_sequence']:
        if action['building'] != 'No-op':
            last_building_turn[action['player']] = action['turn']
    
    print("\nLast Building Turn by Player:")
    for player, turn in last_building_turn.items():
        print(f"  {player}: Turn {turn}")


def analyze_resource_optimization(solution):
    """Analyze resource utilization and efficiency."""
    print("\n" + "=" * 60)
    print("RESOURCE OPTIMIZATION ANALYSIS")
    print("=" * 60)
    
    final_resources = solution['final_resources']
    final_scores = solution['final_scores']
    
    print("Final Resource Efficiency:")
    for player in ['P1', 'P2', 'P3']:
        money = final_resources[player]['money']
        reputation = final_resources[player]['reputation']
        integrated_score = final_scores[player]['integrated_score']
        
        # Calculate resource efficiency (score per resource unit)
        total_resources = money + reputation
        efficiency = integrated_score / total_resources if total_resources > 0 else 0
        
        print(f"  {player}: Money={money:.0f}, Reputation={reputation:.0f}, "
              f"Score={integrated_score:.2f}, Efficiency={efficiency:.3f}")


def analyze_grid_effects(solution):
    """Analyze final grid state and environmental impact."""
    print("\n" + "=" * 60)
    print("GRID STATE ANALYSIS")
    print("=" * 60)
    
    grid_state = solution['final_grid_state']
    
    # Calculate grid statistics
    for metric in ['G', 'V', 'D']:
        grid = np.array(grid_state[metric])
        print(f"\n{metric} (Greenery/Vitality/Density) Statistics:")
        print(f"  Mean: {np.mean(grid):.2f}")
        print(f"  Std:  {np.std(grid):.2f}")
        print(f"  Min:  {np.min(grid):.2f}")
        print(f"  Max:  {np.max(grid):.2f}")
    
    # Environment score
    env_scores = solution['environment_scores']
    print(f"\nEnvironment Score Evolution:")
    print(f"  Initial: {env_scores[0]:.2f}")
    print(f"  Final:   {env_scores[-1]:.2f}")
    print(f"  Change:  {env_scores[-1] - env_scores[0]:.2f}")


def analyze_player_types(solution):
    """Analyze how different player types performed."""
    print("\n" + "=" * 60)
    print("PLAYER TYPE ANALYSIS")
    print("=" * 60)
    
    player_types = {
        'P1': {'type': 'Altruistic', 'alpha': 0.2, 'beta': 0.8},
        'P2': {'type': 'Balanced', 'alpha': 0.5, 'beta': 0.5},
        'P3': {'type': 'InterestDriven', 'alpha': 0.8, 'beta': 0.2}
    }
    
    final_scores = solution['final_scores']
    env_score = solution['environment_scores'][-1]
    
    for player in ['P1', 'P2', 'P3']:
        player_info = player_types[player]
        scores = final_scores[player]
        
        # Calculate expected vs actual integrated score
        alpha, beta = player_info['alpha'], player_info['beta']
        expected_integrated = alpha * scores['self_score'] + beta * env_score
        actual_integrated = scores['integrated_score']
        
        print(f"\n{player} ({player_info['type']}):")
        print(f"  α={alpha}, β={beta}")
        print(f"  Self Score: {scores['self_score']:.2f}")
        print(f"  Expected Integrated: {expected_integrated:.2f}")
        print(f"  Actual Integrated: {actual_integrated:.2f}")
        print(f"  Difference: {actual_integrated - expected_integrated:.2f}")


def analyze_convergence_benchmark(solution):
    """Provide benchmarks for RL convergence analysis."""
    print("\n" + "=" * 60)
    print("RL CONVERGENCE BENCHMARKS")
    print("=" * 60)
    
    total_integrated_score = solution['objective_value']
    final_scores = solution['final_scores']
    
    print(f"Theoretical Optimal Total Integrated Score: {total_integrated_score:.2f}")
    
    individual_scores = [final_scores[player]['integrated_score'] for player in ['P1', 'P2', 'P3']]
    
    print(f"\nIndividual Optimal Scores:")
    for i, player in enumerate(['P1', 'P2', 'P3']):
        score = individual_scores[i]
        print(f"  {player}: {score:.2f}")
    
    print(f"\nBenchmark Thresholds for RL Training:")
    print(f"  95% of optimal: {0.95 * total_integrated_score:.2f}")
    print(f"  90% of optimal: {0.90 * total_integrated_score:.2f}")
    print(f"  85% of optimal: {0.85 * total_integrated_score:.2f}")
    
    print(f"\nRecommended RL Training Targets:")
    print(f"  - Total integrated score should exceed {0.90 * total_integrated_score:.2f}")
    print(f"  - Individual scores should be within 10% of optimal")
    print(f"  - Environment score should be close to {solution['environment_scores'][-1]:.2f}")


def main():
    """Main analysis function."""
    print("SimCity Optimal Solution Analysis")
    print("=" * 50)
    
    try:
        solution = load_solution()
        
        analyze_building_strategy(solution)
        analyze_resource_optimization(solution)
        analyze_grid_effects(solution)
        analyze_player_types(solution)
        analyze_convergence_benchmark(solution)
        
        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETE")
        print("=" * 60)
        
    except FileNotFoundError:
        print("Error: simcity_optimal_solution.json not found.")
        print("Please run optimization_solver.py first.")


if __name__ == "__main__":
    main() 