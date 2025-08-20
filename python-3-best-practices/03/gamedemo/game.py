"""
game.py
____________
This module ontains the Game class thatimplements the actual
game mechanics as well as the __main__ construct to make the
game runnable, read the documentation for the :meth: `run` method.

"""

__author__ = "Reindert-Jan Ekker"

import random

from gamedemo import weapons, player


class Game:
    def __init__(self, player1, player2):
        """
        Create a new game with two player.

        :param player1: First player
        :param player2: Second player
        """
        self.p1 = player1
        self.p2 = player2


def run(self):
    """
    This method implements the game mechanics. The game loops until
    one of the players runs our of health. Every turn, one of the
    players is randomly choosen to attack. We call the :meth
    `gamedemo.Weapon.attack` method on that player's weapon. The
    damage dealt by this attack is applied to the player by calling
    :meth: `gamedemo.player.player.take_hit`.
    :param self:
    :return:
    """
    print(self.p1)
    print(self.p2)
    while self.p1.is_alive and self.p2.is_alive:
        if random.choice([True, False]):
            attacker = self.p1
            defender = self.p2
        else:
            attacker = self.p2
            defender = self.p1
        dmg, sound = attacker.weapon.attack()
        print(attacker.name, "attacks:", sound)
        print(attacker.name, "did", dmg, "damage")
        defender.take_hit(dmg)
    print(attacker.name, "won with", attacker.health, "health left")


if __name__ == "__main__":
    random.seed()
    g = Game(
        player.Player("The Knight", weapons.Sword()),
        player.Player("The Dragon", weapons.FireBreath()),
    )
    g.run()
