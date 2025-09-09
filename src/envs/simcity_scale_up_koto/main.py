# main.py

from environment import SimCityEnv
from players import BalancedPlayer
from utils.logging import get_logger
logger = get_logger(log_file_path="simulation.log")

def main():
    players = {
        "P1": BalancedPlayer("P1"),
        "P2": BalancedPlayer("P2"),
        "P3": BalancedPlayer("P3"),
    }

    env = SimCityEnv()

    env.players = players

    env.run_game()

    print("Game Over!")


if __name__ == "__main__":
    main()
