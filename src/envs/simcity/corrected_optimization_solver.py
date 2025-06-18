"""
Corrected Optimization Solver for Basic SimCity Multi-Agent Environment

This module implements a corrected mathematical optimization model that matches
the actual BASIC training environment reward calculation, including:
- Utility rewards given EVERY TURN for all existing buildings belonging to current player
- Cumulative self score calculation
- Proper individual reward formula: alpha * delta + beta * integrated_score
- Sum of all agents' episode returns as the target 6500 metric

Author: AI Assistant
Date: 2024
"""

import numpy as np
import json
from itertools import product
from typing import Dict, List, Tuple, Optional
import copy

# Configuration constants for the BASIC SimCity environment (from config.py)
BUILDING_TYPES = ['Park', 'House', 'Shop']

BUILDING_COSTS = {
    'Park': {'money': 5, 'reputation': 15},
    'House': {'money': 10, 'reputation': 10},
    'Shop': {'money': 15, 'reputation': 5}
}

BUILDING_UTILITIES = {
    'Park': {'money': -1, 'reputation': 4},
    'House': {'money': 2, 'reputation': 1},
    'Shop': {'money': 3, 'reputation': 0}
}

BUILDING_EFFECTS = {
    'Park': {
        'G': 30, 'V': -30, 'D': 0,
        'neighbors': {'G': 10, 'V': -10, 'D': 0}
    },
    'House': {
        'G': -30, 'V': 0, 'D': 30,
        'neighbors': {'G': -10, 'V': 0, 'D': 10}
    },
    'Shop': {
        'G': 0, 'V': 30, 'D': -30,
        'neighbors': {'G': 0, 'V': 10, 'D': -10}
    }
}

class BasicSimCityOptimizer:
    """
    Corrected optimization model that matches the actual BASIC SimCity training environment.
    """
    
    def __init__(self, grid_size: int = 4, max_turns: int = 16, 
                 initial_money: int = 50, initial_reputation: int = 50,
                 reward_alpha: float = 0.5, reward_beta: float = 0.5):
        """Initialize the corrected optimization model for BASIC SimCity."""
        
        self.grid_size = grid_size
        self.max_turns = max_turns
        self.initial_money = initial_money
        self.initial_reputation = initial_reputation
        self.reward_alpha = reward_alpha
        self.reward_beta = reward_beta
        
        # Players and their types (from environment.py)
        self.players = ['P1', 'P2', 'P3']
        self.player_types = {
            'P1': {'type': 'Altruistic', 'alpha': 0.2, 'beta': 0.8},
            'P2': {'type': 'Balanced', 'alpha': 0.5, 'beta': 0.5},
            'P3': {'type': 'InterestDriven', 'alpha': 0.8, 'beta': 0.2}
        }
        
        # Buildings (from config.py)
        self.buildings = BUILDING_TYPES
        self.costs = BUILDING_COSTS
        self.utilities = BUILDING_UTILITIES
        self.effects = BUILDING_EFFECTS
        
        # Initial grid values (from environment.py reset())
        self.initial_G = 15
        self.initial_V = 20
        self.initial_D = 30
        
    def solve_basic_optimization(self) -> Dict:
        """
        Solve using a simulation approach that matches the BASIC training environment rewards.
        """
        
        print("Solving BASIC SimCity optimization problem (simulation approach)...")
        print(f"Environment: {self.grid_size}x{self.grid_size} grid, {len(self.players)} players")
        print(f"Starting resources: {self.initial_money} money, {self.initial_reputation} reputation")
        print(f"Target: Maximize COMMON REWARD (sum of all integrated scores)")
        
        best_score = float('-inf')
        best_configuration = None
        
        # Try different building placement strategies
        strategies = [
            'maximize_common_reward',
            'aggressive_building_strategy',
            'balanced_long_term_strategy',
            'mixed_utility_strategy',
            'early_building_strategy',
            'balanced_grid_strategy'
        ]
        
        for strategy in strategies:
            print(f"  Evaluating strategy: {strategy}")
            score, config = self._evaluate_basic_strategy(strategy)
            print(f"    Final Common Reward: {score:.0f}")
            if score > best_score:
                best_score = score
                best_configuration = config
                
        return {
            'optimal_common_reward': best_score,
            'optimal_configuration': best_configuration,
            'strategy_used': 'basic_simulation'
        }
    
    def _evaluate_basic_strategy(self, strategy: str) -> Tuple[float, Dict]:
        """Evaluate a strategy using BASIC SimCity environment logic."""
        
        # Initialize game state exactly like environment.py
        money = {player: self.initial_money for player in self.players}
        reputation = {player: self.initial_reputation for player in self.players}
        
        # Initialize grid exactly like environment.py reset()
        grid_G = np.full((self.grid_size, self.grid_size), self.initial_G, dtype=float)
        grid_V = np.full((self.grid_size, self.grid_size), self.initial_V, dtype=float) 
        grid_D = np.full((self.grid_size, self.grid_size), self.initial_D, dtype=float)
        
        buildings_placed = np.full((self.grid_size, self.grid_size), None)
        builders = np.full((self.grid_size, self.grid_size), -1)
        
        # Track scores exactly like environment.py
        self_score = {player: 0 for player in self.players}
        integrated_score = {player: 0 for player in self.players}
        previous_integrated_score = {player: 0 for player in self.players}
        episode_return = {player: 0 for player in self.players}
        
        actions_taken = []
        common_reward_progression = []
        
        # Simulate game for exactly 100 steps like the training environment
        for turn in range(100):
            current_player = self.players[turn % len(self.players)]
            
            # Determine best action for current player
            best_action = self._get_basic_best_action(
                current_player, strategy, money, reputation,
                grid_G, grid_V, grid_D, buildings_placed, turn
            )
            
            if best_action is not None:
                building, i, j = best_action
                
                # Check if action is valid
                if (buildings_placed[i, j] is None and 
                    money[current_player] >= self.costs[building]['money'] and
                    reputation[current_player] >= self.costs[building]['reputation']):
                    
                    # Apply action - deduct building costs
                    money[current_player] -= self.costs[building]['money']
                    reputation[current_player] -= self.costs[building]['reputation']
                    
                    buildings_placed[i, j] = building
                    builders[i, j] = self.players.index(current_player)
                    
                    # Update grid effects exactly like environment.py
                    self._apply_basic_building_effects(building, i, j, grid_G, grid_V, grid_D)
                    
                    # Immediate reward from building
                    utility = self.utilities[building]
                    immediate_reward = utility['money'] + utility['reputation']
                    
                    actions_taken.append({
                        'turn': turn,
                        'player': current_player,
                        'building': building,
                        'position': (i, j),
                        'immediate_reward': immediate_reward
                    })
            
            # CRITICAL: Apply utility effects for ALL existing buildings owned by current player
            # This matches the environment.py logic exactly
            for i, j in product(range(self.grid_size), range(self.grid_size)):
                if buildings_placed[i, j] is not None and builders[i, j] == self.players.index(current_player):
                    building = buildings_placed[i, j]
                    utility = self.utilities[building]
                    money[current_player] += utility['money']
                    reputation[current_player] += utility['reputation']
                    self_score[current_player] += utility['money'] + utility['reputation']
            
            # Calculate environment score exactly like environment.py
            env_score = (np.mean(grid_G) + np.mean(grid_V) + np.mean(grid_D)) / 3
            
            # Calculate integrated score using player-specific alpha/beta
            alpha = self.player_types[current_player]['alpha']
            beta = self.player_types[current_player]['beta']
            integrated_score[current_player] = alpha * self_score[current_player] + beta * env_score
            
            # Calculate individual reward using the training environment formula
            # reward = alpha * delta + beta * integrated_score (from compute_individual_reward)
            current_integrated = integrated_score[current_player]
            previous_integrated = previous_integrated_score[current_player]
            delta = current_integrated - previous_integrated
            individual_reward = self.reward_alpha * delta + self.reward_beta * current_integrated
            
            episode_return[current_player] += individual_reward
            previous_integrated_score[current_player] = current_integrated
            
            # Calculate common reward value (the TARGET 6500 metric!)
            # This is the sum of all players' integrated scores
            common_reward_value = sum(integrated_score.values())
            common_reward_progression.append(common_reward_value)
            
            # IMPORTANT: Unlike the environment that stops when board is full,
            # the training wrapper continues for the full 100 steps accumulating utility rewards
            # So we DON'T break early - we continue for all 100 turns
        
        # Calculate final scores
        final_env_score = (np.mean(grid_G) + np.mean(grid_V) + np.mean(grid_D)) / 3
        
        # Update final integrated scores for all players
        for player in self.players:
            alpha = self.player_types[player]['alpha']
            beta = self.player_types[player]['beta']
            integrated_score[player] = alpha * self_score[player] + beta * final_env_score
        
        # The FINAL COMMON REWARD is what we want to maximize (6500 target)
        final_common_reward = sum(integrated_score.values())
        
        # The target 6500 is the sum of all agents' episode returns
        total_episode_returns = sum(episode_return.values())
        
        final_scores = {}
        for player in self.players:
            alpha = self.player_types[player]['alpha']
            beta = self.player_types[player]['beta']
            final_integrated = alpha * self_score[player] + beta * final_env_score
            
            final_scores[player] = {
                'money': money[player],
                'reputation': reputation[player],
                'self_score': self_score[player],
                'integrated_score': final_integrated,
                'episode_return': episode_return[player]
            }
        
        configuration = {
            'actions': actions_taken,
            'final_scores': final_scores,
            'final_grid': {
                'G': grid_G.tolist(),
                'V': grid_V.tolist(), 
                'D': grid_D.tolist(),
                'buildings': buildings_placed.tolist(),
                'builders': builders.tolist()
            },
            'environment_score': final_env_score,
            'final_common_reward': final_common_reward,
            'total_episode_returns': total_episode_returns,
            'common_reward_progression': common_reward_progression
        }
        
        # Return the COMMON REWARD as the score to maximize (not episode returns)
        return final_common_reward, configuration
    
    def _get_basic_best_action(self, player: str, strategy: str, money: Dict, reputation: Dict,
                              grid_G: np.ndarray, grid_V: np.ndarray, grid_D: np.ndarray,
                              buildings_placed: np.ndarray, turn: int = 0) -> Optional[Tuple[str, int, int]]:
        """Get the best action for a player given the strategy."""
        
        valid_actions = []
        
        # Find all valid building placements
        for building in self.buildings:
            for i, j in product(range(self.grid_size), range(self.grid_size)):
                if (buildings_placed[i, j] is None and
                    money[player] >= self.costs[building]['money'] and
                    reputation[player] >= self.costs[building]['reputation']):
                    valid_actions.append((building, i, j))
        
        if not valid_actions:
            return None
        
        if strategy == 'maximize_common_reward':
            # Prioritize buildings that will generate most utility over remaining game
            best_action = None
            best_value = float('-inf')
            
            remaining_turns = max(1, 100 - turn)  # Use full simulation length
            
            for building, i, j in valid_actions:
                utility = self.utilities[building]['money'] + self.utilities[building]['reputation']
                # Estimate total utility over remaining turns
                estimated_total_utility = utility * remaining_turns
                
                # Add immediate reward value
                immediate_value = utility
                total_estimated_value = estimated_total_utility + immediate_value
                
                if total_estimated_value > best_value:
                    best_value = total_estimated_value
                    best_action = (building, i, j)
            
            return best_action
            
        elif strategy == 'aggressive_building_strategy':
            # Build as much as possible - prefer buildings with positive utility
            positive_buildings = ['House', 'Shop']  # These have positive total utility
            for building in positive_buildings:
                for b, i, j in valid_actions:
                    if b == building:
                        return (b, i, j)
            # Fall back to any available action
            return valid_actions[0] if valid_actions else None
            
        elif strategy == 'balanced_long_term_strategy':
            # Balance between immediate utility and environment effects
            best_action = None
            best_score = float('-inf')
            
            for building, i, j in valid_actions:
                utility = self.utilities[building]['money'] + self.utilities[building]['reputation']
                
                # Consider both immediate utility and environment impact
                effects = self.effects[building]
                env_impact = abs(effects['G']) + abs(effects['V']) + abs(effects['D'])
                
                # Score based on utility and environmental impact
                score = utility * 2 + env_impact * 0.5  # Weight utility higher
                
                if score > best_score:
                    best_score = score
                    best_action = (building, i, j)
                    
            return best_action if best_action else valid_actions[0]
            
        elif strategy == 'mixed_utility_strategy':
            # Prefer positive utility buildings
            positive_utility_actions = []
            for building, i, j in valid_actions:
                utility = self.utilities[building]['money'] + self.utilities[building]['reputation']
                if utility > 0:
                    positive_utility_actions.append((building, i, j, utility))
            
            if positive_utility_actions:
                # Sort by utility and pick best
                positive_utility_actions.sort(key=lambda x: x[3], reverse=True)
                return positive_utility_actions[0][:3]
            else:
                return valid_actions[0] if valid_actions else None
            
        elif strategy == 'early_building_strategy':
            # Build as many as possible early
            if turn < self.max_turns // 2:
                # Prefer cheaper buildings early on
                cheap_actions = [(b, i, j) for b, i, j in valid_actions 
                               if self.costs[b]['money'] + self.costs[b]['reputation'] <= 15]
                if cheap_actions:
                    return cheap_actions[0]
            return valid_actions[0] if valid_actions else None
            
        elif strategy == 'balanced_grid_strategy':
            # Try to balance grid effects
            # Prefer buildings that don't make grid too extreme
            best_action = None
            best_balance = float('-inf')
            
            for building, i, j in valid_actions:
                # Estimate grid balance after this building
                temp_G = np.mean(grid_G) 
                temp_V = np.mean(grid_V)
                temp_D = np.mean(grid_D)
                
                # Add building effects
                effects = self.effects[building]
                temp_G += effects['G'] / (self.grid_size * self.grid_size)
                temp_V += effects['V'] / (self.grid_size * self.grid_size)
                temp_D += effects['D'] / (self.grid_size * self.grid_size)
                
                # Calculate balance (prefer values close to each other)
                balance = -(abs(temp_G - temp_V) + abs(temp_V - temp_D) + abs(temp_G - temp_D))
                
                if balance > best_balance:
                    best_balance = balance
                    best_action = (building, i, j)
                    
            return best_action if best_action else valid_actions[0]
        
        else:  # Default fallback
            return valid_actions[0] if valid_actions else None
    
    def _apply_basic_building_effects(self, building: str, i: int, j: int,
                                    grid_G: np.ndarray, grid_V: np.ndarray, grid_D: np.ndarray):
        """Apply building effects to the grid exactly like environment.py."""
        
        effects = self.effects[building]
        
        # Direct effects on the cell
        grid_G[i, j] += effects['G']
        grid_V[i, j] += effects['V']
        grid_D[i, j] += effects['D']
        
        # Neighbor effects (8-directional as in environment.py)
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (1, 1), (-1, -1), (1, -1), (-1, 1)]:
            ni, nj = i + dx, j + dy
            if 0 <= ni < self.grid_size and 0 <= nj < self.grid_size:
                grid_G[ni, nj] += effects['neighbors']['G']
                grid_V[ni, nj] += effects['neighbors']['V']
                grid_D[ni, nj] += effects['neighbors']['D']
     
    def print_solution(self, solution: Dict):
        """Print a human-readable version of the corrected solution."""
        
        print("=" * 70)
        print("BASIC SIMCITY OPTIMIZATION SOLUTION")
        print("=" * 70)
        
        print(f"\nOptimal COMMON REWARD: {solution['optimal_common_reward']:.0f}")
        
        config = solution['optimal_configuration']
        
        print("\nBuilding Sequence:")
        print("-" * 50)
        for action in config['actions']:
            print(f"Turn {action['turn']:2d}: {action['player']} -> {action['building']} at {action['position']} (reward: {action['immediate_reward']})")
        
        print(f"\nBuilding Summary:")
        print("-" * 50)
        building_counts = {}
        for action in config['actions']:
            building = action['building']
            building_counts[building] = building_counts.get(building, 0) + 1
        
        for building, count in building_counts.items():
            print(f"  {building}: {count} buildings")
        print(f"  Total buildings: {sum(building_counts.values())}")
        
        print("\nFinal Scores:")
        print("-" * 50)
        for player in self.players:
            scores = config['final_scores'][player]
            player_type = self.player_types[player]['type']
            print(f"{player} ({player_type}):")
            print(f"  Money: {scores['money']:.0f}, Reputation: {scores['reputation']:.0f}")
            print(f"  Self Score: {scores['self_score']:.2f}")
            print(f"  Integrated Score: {scores['integrated_score']:.2f}")
            print(f"  Episode Return: {scores['episode_return']:.2f}")
        
        print(f"\nFinal Environment Score: {config['environment_score']:.2f}")
        print(f"Final Common Reward (Target Metric): {config['final_common_reward']:.0f}")
        print(f"Total Episode Returns (Sum): {config['total_episode_returns']:.0f}")
        
        # Show common reward progression (first 10 and last 10 turns)
        if config['common_reward_progression']:
            print(f"\nCommon Reward Progression (first 10 and last 10 turns):")
            print("-" * 50)
            progression = config['common_reward_progression']
            if len(progression) <= 20:
                for i, reward in enumerate(progression):
                    print(f"Turn {i:2d}: {reward:.1f}")
            else:
                for i in range(10):
                    print(f"Turn {i:2d}: {progression[i]:.1f}")
                print("...")
                for i in range(max(10, len(progression)-10), len(progression)):
                    print(f"Turn {i:2d}: {progression[i]:.1f}")
        
        print("\nComparison with Your Training Results:")
        print("-" * 50)
        expected_reward = config['final_common_reward']
        print(f"Theoretical Optimal Common Reward: {expected_reward:.0f}")
        print(f"Your RL Training Result (common reward): ~6500")
        
        if expected_reward > 0:
            performance_ratio = 6500 / expected_reward
            print(f"Performance Ratio: {performance_ratio:.1%} of optimal")
            
            if performance_ratio >= 0.85:
                print("✅ Excellent! Your RL training is achieving near-optimal performance!")
            elif performance_ratio >= 0.70:
                print("✅ Good! Your RL training is performing well!")
            else:
                print("⚠️  There may be room for improvement in your RL training.")
        else:
            print("⚠️  Still investigating reward calculation...")


def main():
    """Main function to run the corrected optimization for BASIC SimCity."""
    
    print("Basic SimCity Multi-Agent Optimization Solver")
    print("=" * 55)
    print("Matching your ACTUAL basic training environment configuration")
    
    # Create basic optimizer instance with exact environment parameters
    optimizer = BasicSimCityOptimizer(
        grid_size=4,           # 4x4 grid
        max_turns=16,          # 16 turns max (4x4 = 16 cells)
        initial_money=50,      # From environment.py reset()
        initial_reputation=50, # From environment.py reset()
        reward_alpha=0.5,      # From environment.py __init__()
        reward_beta=0.5        # From environment.py __init__()
    )
    
    print("Solving basic SimCity optimization problem...")
    solution = optimizer.solve_basic_optimization()
    
    print("Basic optimization completed!")
    optimizer.print_solution(solution)
    
    # Save solution
    with open('simcity_basic_optimal_solution.json', 'w') as f:
        json.dump(solution, f, indent=2)
    print("\nBasic solution saved to simcity_basic_optimal_solution.json")
    
    return optimizer, solution


if __name__ == "__main__":
    optimizer, solution = main()