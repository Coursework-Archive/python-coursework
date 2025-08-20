"""
    player.py
    _________

    This module contains the Player class that
    represents game characters
"""

__author__ = "Reindert-Jan Ekker"

class Player:
    """
    The Player class represents thee characters in the
    game.

    :ivar health: The current health of the character.
    Starts at 100. Once it reaches 0, we're  dead.

    """

    def __init__(self, name, weapon):
        """
        Create a new Player.

        :param name: The name of the Player
        :param weapon: The weapon that this Player uses to fight with
        """
        self.name = name
        self.weapon = weapon
        self.health = 100

    def take_hit(self, damage):
        """
        This method gets called when the player takes a hit from the
        opponent's weapon
        :param damage: The damage dealt. This will be subtracted
        from :attr: `health`.
        :return: The new value of :attr: `health`.
        """
        self.health -= damage
        return self.health

    # @property
    # def is_alive(self):