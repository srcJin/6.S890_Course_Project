"""
Optimization Solver for SimCity Multi-Agent Environment

This module implements a mathematical optimization model to find the theoretical
optimal reward and game configuration for the SimCity multi-agent environment.

The problem is modeled as a Mixed Integer Linear Programming (MILP) problem that
considers all constraints from the original environment:
- Resource limitations for each player
- Spatial constraints (one building per cell)
- Turn order and temporal dynamics
- Building effects on grid indices
- Player-specific utility functions

Author: AI Assistant
Date: 2024
"""

import numpy as np
import pulp
import json
from itertools import product
from typing import Dict, List, Tuple, Optional
import copy

# Configuration constants for the SimCity environment
BUILDING_TYPES = ['Park', 'House', 'Shop']

BUILDING_COSTS = {
    'Park': {'money': 5, 'reputation': 15},
    'House': {'money': 10, 'reputation': 10},
    'Shop': {'money': 15, 'reputation': 5}
}

BUILDING_UTILITIES = {
    'Park': {'money': 1, 'reputation': 3},
    'House': {'money': 2, 'reputation': 2},
    'Shop': {'money': 3, 'reputation': 1}
}

BUILDING_EFFECTS = {
    'Park': {
        'G': 10, 'V': 5, 'D': -5,
        'neighbors': {'G': 2, 'V': 1, 'D': -1}
    },
    'House': {
        'G': -5, 'V': 10, 'D': 5,
        'neighbors': {'G': -1, 'V': 2, 'D': 1}
    },
    'Shop': {
        'G': -10, 'V': -5, 'D': 15,
        'neighbors': {'G': -2, 'V': -1, 'D': 3}
    }
}


class SimCityOptimizer:
    """
    Optimization model for the SimCity multi-agent environment.
    
    This class formulates and solves a MILP to find the theoretical optimal
    solution for maximizing total rewards across all players.
    """
    
    def __init__(self, grid_size: int = 4, max_turns: int = 16, 
                 initial_money: int = 50, initial_reputation: int = 50,
                 reward_alpha: float = 0.5, reward_beta: float = 0.5):
        """
        Initialize the optimization model.
        
        Args:
            grid_size: Size of the square grid (default 4x4)
            max_turns: Maximum number of turns in the game
            initial_money: Starting money for each player
            initial_reputation: Starting reputation for each player
            reward_alpha: Weight for delta in reward calculation
            reward_beta: Weight for integrated score in reward calculation
        """
        self.grid_size = grid_size
        self.max_turns = max_turns
        self.initial_money = initial_money
        self.initial_reputation = initial_reputation
        self.reward_alpha = reward_alpha
        self.reward_beta = reward_beta
        
        # Players and their types
        self.players = ['P1', 'P2', 'P3']
        self.player_types = {
            'P1': {'type': 'Altruistic', 'alpha': 0.2, 'beta': 0.8},
            'P2': {'type': 'Balanced', 'alpha': 0.5, 'beta': 0.5},
            'P3': {'type': 'InterestDriven', 'alpha': 0.8, 'beta': 0.2}
        }
        
        # Buildings
        self.buildings = BUILDING_TYPES
        
        # Building costs (money, reputation)
        self.costs = BUILDING_COSTS
        
        # Building utilities (ongoing generation)
        self.utilities = BUILDING_UTILITIES
        
        # Building effects on grid indices (G, V, D)
        self.effects = BUILDING_EFFECTS
        
        # Initial grid values
        self.initial_G = 15
        self.initial_V = 20
        self.initial_D = 30
        
        # Initialize optimization problem
        self.prob = None
        self.variables = {}
        self.solution = None
        
    def create_variables(self):
        """Create decision variables for the optimization model."""
        
        # Decision variables: x[player, turn, building, i, j]
        # Binary variable: 1 if player builds building at position (i,j) on turn t
        self.variables['build'] = pulp.LpVariable.dicts(
            "build",
            (self.players, range(self.max_turns), self.buildings, 
             range(self.grid_size), range(self.grid_size)),
            cat='Binary'
        )
        
        # Resource variables: money and reputation for each player at each turn
        self.variables['money'] = pulp.LpVariable.dicts(
            "money",
            (self.players, range(self.max_turns + 1)),
            lowBound=0,
            cat='Integer'
        )
        
        self.variables['reputation'] = pulp.LpVariable.dicts(
            "reputation", 
            (self.players, range(self.max_turns + 1)),
            lowBound=0,
            cat='Integer'
        )
        
        # Grid state variables: G, V, D values at each cell for each turn
        self.variables['grid_G'] = pulp.LpVariable.dicts(
            "grid_G",
            (range(self.max_turns + 1), range(self.grid_size), range(self.grid_size)),
            lowBound=0,
            upBound=200,
            cat='Integer'
        )
        
        self.variables['grid_V'] = pulp.LpVariable.dicts(
            "grid_V",
            (range(self.max_turns + 1), range(self.grid_size), range(self.grid_size)),
            lowBound=0,
            upBound=200,
            cat='Integer'
        )
        
        self.variables['grid_D'] = pulp.LpVariable.dicts(
            "grid_D",
            (range(self.max_turns + 1), range(self.grid_size), range(self.grid_size)),
            lowBound=0,
            upBound=200,
            cat='Integer'
        )
        
        # Environment score at each turn
        self.variables['env_score'] = pulp.LpVariable.dicts(
            "env_score",
            range(self.max_turns + 1),
            lowBound=0,
            upBound=200,
            cat='Continuous'
        )
        
        # Self score and integrated score for each player at each turn
        self.variables['self_score'] = pulp.LpVariable.dicts(
            "self_score",
            (self.players, range(self.max_turns + 1)),
            lowBound=0,
            cat='Continuous'
        )
        
        self.variables['integrated_score'] = pulp.LpVariable.dicts(
            "integrated_score",
            (self.players, range(self.max_turns + 1)),
            cat='Continuous'
        )
        
        # Auxiliary variables for turn order (which player plays at each turn)
        self.variables['player_turn'] = pulp.LpVariable.dicts(
            "player_turn",
            (range(self.max_turns), self.players),
            cat='Binary'
        )
        
    def add_constraints(self):
        """Add all constraints to the optimization model."""
        
        # Initial conditions
        self._add_initial_conditions()
        
        # Turn order constraints
        self._add_turn_order_constraints()
        
        # Building placement constraints
        self._add_building_constraints()
        
        # Resource dynamics constraints
        self._add_resource_constraints()
        
        # Grid dynamics constraints
        self._add_grid_constraints()
        
        # Score calculation constraints
        self._add_score_constraints()
        
    def _add_initial_conditions(self):
        """Add initial state constraints."""
        
        # Initial resources
        for player in self.players:
            self.prob += self.variables['money'][player][0] == self.initial_money
            self.prob += self.variables['reputation'][player][0] == self.initial_reputation
            self.prob += self.variables['self_score'][player][0] == (
                0.5 * self.initial_money + 0.5 * self.initial_reputation
            )
             
        # Initial grid values
        for i, j in product(range(self.grid_size), range(self.grid_size)):
            self.prob += self.variables['grid_G'][0][i][j] == self.initial_G
            self.prob += self.variables['grid_V'][0][i][j] == self.initial_V
            self.prob += self.variables['grid_D'][0][i][j] == self.initial_D
        
        # Initial environment score
        self.prob += self.variables['env_score'][0] == (
            (self.initial_G + self.initial_V + self.initial_D) / 3
        )
        
        # Initial integrated scores
        for player in self.players:
            alpha = self.player_types[player]['alpha']
            beta = self.player_types[player]['beta']
            self.prob += self.variables['integrated_score'][player][0] == (
                alpha * self.variables['self_score'][player][0] + 
                beta * self.variables['env_score'][0]
            )
    
    def _add_turn_order_constraints(self):
        """Add constraints for turn order (players take turns in order P1, P2, P3)."""
        
        for t in range(self.max_turns):
            # Exactly one player plays per turn
            self.prob += pulp.lpSum([
                self.variables['player_turn'][t][player] for player in self.players
            ]) == 1
            
            # Players take turns in order P1, P2, P3, P1, P2, P3, ...
            player_index = t % len(self.players)
            current_player = self.players[player_index]
            self.prob += self.variables['player_turn'][t][current_player] == 1
    
    def _add_building_constraints(self):
        """Add constraints for building placement."""
        
        # At most one building per cell across all turns
        for i, j in product(range(self.grid_size), range(self.grid_size)):
            self.prob += pulp.lpSum([
                self.variables['build'][player][t][building][i][j]
                for player in self.players
                for t in range(self.max_turns)
                for building in self.buildings
            ]) <= 1
        
        # At most one action per player per turn
        for player in self.players:
            for t in range(self.max_turns):
                self.prob += pulp.lpSum([
                    self.variables['build'][player][t][building][i][j]
                    for building in self.buildings
                    for i, j in product(range(self.grid_size), range(self.grid_size))
                ]) <= self.variables['player_turn'][t][player]
    
    def _add_resource_constraints(self):
        """Add constraints for resource dynamics."""
        
        for player in self.players:
            for t in range(self.max_turns):
                # Money constraint
                money_spent = pulp.lpSum([
                    self.costs[building]['money'] * self.variables['build'][player][t][building][i][j]
                    for building in self.buildings
                    for i, j in product(range(self.grid_size), range(self.grid_size))
                ])
                
                money_earned = pulp.lpSum([
                    self.utilities[building]['money'] * self.variables['build'][player][tau][building][i][j]
                    for building in self.buildings
                    for i, j in product(range(self.grid_size), range(self.grid_size))
                    for tau in range(t)  # Buildings built in previous turns
                ])
                
                self.prob += (self.variables['money'][player][t+1] == 
                             self.variables['money'][player][t] - money_spent + money_earned)
                
                # Reputation constraint
                reputation_spent = pulp.lpSum([
                    self.costs[building]['reputation'] * self.variables['build'][player][t][building][i][j]
                    for building in self.buildings
                    for i, j in product(range(self.grid_size), range(self.grid_size))
                ])
                
                reputation_earned = pulp.lpSum([
                    self.utilities[building]['reputation'] * self.variables['build'][player][tau][building][i][j]
                    for building in self.buildings
                    for i, j in product(range(self.grid_size), range(self.grid_size))
                    for tau in range(t)  # Buildings built in previous turns
                ])
                
                self.prob += (self.variables['reputation'][player][t+1] == 
                             self.variables['reputation'][player][t] - reputation_spent + reputation_earned)
                
                # Resource availability constraints (can only build if have enough resources)
                for building in self.buildings:
                    for i, j in product(range(self.grid_size), range(self.grid_size)):
                        # If building this, must have enough money and reputation
                        self.prob += (
                            self.variables['money'][player][t] >= 
                            self.costs[building]['money'] * self.variables['build'][player][t][building][i][j]
                        )
                        self.prob += (
                            self.variables['reputation'][player][t] >= 
                            self.costs[building]['reputation'] * self.variables['build'][player][t][building][i][j]
                        )
    
    def _add_grid_constraints(self):
        """Add constraints for grid dynamics (G, V, D values)."""
        
        for t in range(self.max_turns):
            for i, j in product(range(self.grid_size), range(self.grid_size)):
                # Grid updates based on buildings placed this turn
                
                # Direct effects on the cell where building is placed
                direct_G_effect = pulp.lpSum([
                    self.effects[building]['G'] * self.variables['build'][player][t][building][i][j]
                    for player in self.players
                    for building in self.buildings
                ])
                
                direct_V_effect = pulp.lpSum([
                    self.effects[building]['V'] * self.variables['build'][player][t][building][i][j]
                    for player in self.players
                    for building in self.buildings
                ])
                
                direct_D_effect = pulp.lpSum([
                    self.effects[building]['D'] * self.variables['build'][player][t][building][i][j]
                    for player in self.players
                    for building in self.buildings
                ])
                
                # Neighbor effects from buildings placed this turn
                neighbor_G_effect = 0
                neighbor_V_effect = 0
                neighbor_D_effect = 0
                
                # Check all possible neighbor positions
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == 0 and dj == 0:  # Skip the cell itself
                            continue
                        ni, nj = i + di, j + dj
                        if 0 <= ni < self.grid_size and 0 <= nj < self.grid_size:
                            neighbor_G_effect += pulp.lpSum([
                                self.effects[building]['neighbors']['G'] * 
                                self.variables['build'][player][t][building][ni][nj]
                                for player in self.players
                                for building in self.buildings
                            ])
                            neighbor_V_effect += pulp.lpSum([
                                self.effects[building]['neighbors']['V'] * 
                                self.variables['build'][player][t][building][ni][nj]
                                for player in self.players
                                for building in self.buildings
                            ])
                            neighbor_D_effect += pulp.lpSum([
                                self.effects[building]['neighbors']['D'] * 
                                self.variables['build'][player][t][building][ni][nj]
                                for player in self.players
                                for building in self.buildings
                            ])
                
                # Update grid values
                self.prob += (self.variables['grid_G'][t+1][i][j] == 
                             self.variables['grid_G'][t][i][j] + direct_G_effect + neighbor_G_effect)
                
                self.prob += (self.variables['grid_V'][t+1][i][j] == 
                             self.variables['grid_V'][t][i][j] + direct_V_effect + neighbor_V_effect)
                
                self.prob += (self.variables['grid_D'][t+1][i][j] == 
                             self.variables['grid_D'][t][i][j] + direct_D_effect + neighbor_D_effect)
            
            # Environment score calculation
            total_cells = self.grid_size * self.grid_size
            avg_G = pulp.lpSum([
                self.variables['grid_G'][t+1][i][j] 
                for i, j in product(range(self.grid_size), range(self.grid_size))
            ]) / total_cells
            
            avg_V = pulp.lpSum([
                self.variables['grid_V'][t+1][i][j] 
                for i, j in product(range(self.grid_size), range(self.grid_size))
            ]) / total_cells
            
            avg_D = pulp.lpSum([
                self.variables['grid_D'][t+1][i][j] 
                for i, j in product(range(self.grid_size), range(self.grid_size))
            ]) / total_cells
            
            self.prob += self.variables['env_score'][t+1] == (avg_G + avg_V + avg_D) / 3
    
    def _add_score_constraints(self):
        """Add constraints for score calculations."""
        
        for player in self.players:
            for t in range(self.max_turns):
                # Self score = 0.5 * money + 0.5 * reputation
                self.prob += (self.variables['self_score'][player][t+1] == 
                             0.5 * self.variables['money'][player][t+1] + 
                             0.5 * self.variables['reputation'][player][t+1])
                
                # Integrated score = alpha * self_score + beta * env_score
                alpha = self.player_types[player]['alpha']
                beta = self.player_types[player]['beta']
                self.prob += (self.variables['integrated_score'][player][t+1] == 
                             alpha * self.variables['self_score'][player][t+1] + 
                             beta * self.variables['env_score'][t+1])
    
    def create_objective(self, objective_type: str = 'total_integrated'):
        """
        Create the objective function to maximize.
        
        Args:
            objective_type: Type of objective ('total_integrated', 'common_reward', 'individual_sum')
        """
        
        if objective_type == 'total_integrated':
            # Maximize sum of final integrated scores
            objective = pulp.lpSum([
                self.variables['integrated_score'][player][self.max_turns]
                for player in self.players
            ])
        elif objective_type == 'common_reward':
            # Maximize common reward (sum of integrated scores)
            objective = pulp.lpSum([
                self.variables['integrated_score'][player][self.max_turns]
                for player in self.players
            ])
        elif objective_type == 'individual_sum':
            # Maximize sum of individual rewards (simplified)
            objective = pulp.lpSum([
                self.reward_alpha * (
                    self.variables['integrated_score'][player][self.max_turns] - 
                    self.variables['integrated_score'][player][0]
                ) + self.reward_beta * self.variables['integrated_score'][player][self.max_turns]
                for player in self.players
            ])
        else:
            raise ValueError(f"Unknown objective type: {objective_type}")
        
        self.prob += objective
        
    def solve(self, objective_type: str = 'total_integrated', solver_name: str = 'PULP_CBC_CMD') -> bool:
        """
        Solve the optimization problem.
        
        Args:
            objective_type: Type of objective to maximize
            solver_name: Name of the solver to use
            
        Returns:
            True if optimal solution found, False otherwise
        """
        
        print(f"Formulating MILP with {self.grid_size}x{self.grid_size} grid, {len(self.players)} players...")
        
        # Create the problem
        self.prob = pulp.LpProblem("SimCity_Optimization", pulp.LpMaximize)
        
        # Create variables
        print("Creating decision variables...")
        self.create_variables()
        
        # Add constraints
        print("Adding constraints...")
        self.add_constraints()
        
        # Create objective
        print(f"Setting objective: {objective_type}")
        self.create_objective(objective_type)
        
        # Solve
        print(f"Solving with {solver_name}...")
        solver = pulp.getSolver(solver_name)
        status = self.prob.solve(solver)
        
        # Check if solution is optimal
        if status == pulp.LpStatusOptimal:
            print("Optimal solution found!")
            self.solution = self._extract_solution()
            return True
        else:
            print(f"Optimization failed with status: {pulp.LpStatus[status]}")
            return False
    
    def _extract_solution(self) -> Dict:
        """Extract the solution from the solved optimization problem."""
        
        solution = {
            'status': 'optimal',
            'objective_value': pulp.value(self.prob.objective),
            'building_sequence': [],
            'final_scores': {},
            'final_resources': {},
            'final_grid_state': {},
            'environment_scores': []
        }
        
        # Extract building sequence
        for t in range(self.max_turns):
            turn_actions = []
            for player in self.players:
                if pulp.value(self.variables['player_turn'][t][player]) == 1:
                    action_found = False
                    for building in self.buildings:
                        for i, j in product(range(self.grid_size), range(self.grid_size)):
                            if pulp.value(self.variables['build'][player][t][building][i][j]) == 1:
                                turn_actions.append({
                                    'turn': t,
                                    'player': player,
                                    'building': building,
                                    'position': (i, j)
                                })
                                action_found = True
                                break
                        if action_found:
                            break
                    if not action_found:
                        turn_actions.append({
                            'turn': t,
                            'player': player,
                            'building': 'No-op',
                            'position': None
                        })
            solution['building_sequence'].extend(turn_actions)
        
        # Extract final scores and resources
        for player in self.players:
            solution['final_scores'][player] = {
                'self_score': pulp.value(self.variables['self_score'][player][self.max_turns]),
                'integrated_score': pulp.value(self.variables['integrated_score'][player][self.max_turns])
            }
            solution['final_resources'][player] = {
                'money': pulp.value(self.variables['money'][player][self.max_turns]),
                'reputation': pulp.value(self.variables['reputation'][player][self.max_turns])
            }
        
        # Extract final grid state
        solution['final_grid_state'] = {
            'G': [[pulp.value(self.variables['grid_G'][self.max_turns][i][j]) 
                   for j in range(self.grid_size)] for i in range(self.grid_size)],
            'V': [[pulp.value(self.variables['grid_V'][self.max_turns][i][j]) 
                   for j in range(self.grid_size)] for i in range(self.grid_size)],
            'D': [[pulp.value(self.variables['grid_D'][self.max_turns][i][j]) 
                   for j in range(self.grid_size)] for i in range(self.grid_size)]
        }
        
        # Extract environment scores over time
        for t in range(self.max_turns + 1):
            solution['environment_scores'].append(
                pulp.value(self.variables['env_score'][t])
            )
        
        return solution
     
    def print_solution(self):
        """Print a human-readable version of the solution."""
        
        if self.solution is None:
            print("No solution available. Run solve() first.")
            return
        
        print("=" * 60)
        print("SIMCITY OPTIMIZATION SOLUTION")
        print("=" * 60)
        
        print(f"\nObjective Value: {self.solution['objective_value']:.2f}")
        
        print("\nBuilding Sequence:")
        print("-" * 40)
        for action in self.solution['building_sequence']:
            if action['building'] == 'No-op':
                print(f"Turn {action['turn']:2d}: {action['player']} -> No-op")
            else:
                print(f"Turn {action['turn']:2d}: {action['player']} -> {action['building']} at {action['position']}")
        
        print("\nFinal Scores:")
        print("-" * 40)
        for player in self.players:
            scores = self.solution['final_scores'][player]
            player_type = self.player_types[player]['type']
            print(f"{player} ({player_type}): Self={scores['self_score']:.2f}, Integrated={scores['integrated_score']:.2f}")
        
        print("\nFinal Resources:")
        print("-" * 40)
        for player in self.players:
            resources = self.solution['final_resources'][player]
            print(f"{player}: Money={resources['money']:.0f}, Reputation={resources['reputation']:.0f}")
        
        print(f"\nFinal Environment Score: {self.solution['environment_scores'][-1]:.2f}")
        
        print("\nFinal Grid State (G/V/D):")
        print("-" * 40)
        for attr in ['G', 'V', 'D']:
            print(f"\n{attr} values:")
            grid = self.solution['final_grid_state'][attr]
            for row in grid:
                print("  " + " ".join(f"{val:6.1f}" for val in row))
    
    def save_solution(self, filename: str):
        """Save the solution to a JSON file."""
        
        if self.solution is None:
            print("No solution available. Run solve() first.")
            return
        
        with open(filename, 'w') as f:
            json.dump(self.solution, f, indent=2)
        print(f"Solution saved to {filename}")


def main():
    """Main function to run the optimization."""
    
    print("SimCity Multi-Agent Optimization Solver")
    print("=" * 50)
    
    # Create optimizer instance
    optimizer = SimCityOptimizer(
        grid_size=4,
        max_turns=16,  # Game ends when board is full (16 cells max)
        initial_money=50,
        initial_reputation=50
    )
    
    print("Setting up optimization problem...")
    success = optimizer.solve(objective_type='total_integrated')
    
    if success:
        print("Optimization completed successfully!")
        optimizer.print_solution()
        optimizer.save_solution('simcity_optimal_solution.json')
    else:
        print("Optimization failed!")
    
    return optimizer


if __name__ == "__main__":
    optimizer = main()