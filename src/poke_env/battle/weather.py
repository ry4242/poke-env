"""This module defines the Weather class, which represents a in-battle weather."""

import logging
from enum import Enum, auto
from typing import ClassVar, FrozenSet


class Weather(Enum):
    """Enumeration, represent a non null weather in a battle."""

    UNKNOWN = auto()
    SUNNYDAY = auto()
    RAINDANCE = auto()
    HAIL = auto()
    SNOWSCAPE = SNOW = auto()
    BLOODMOON = auto()
    FOGHORN = auto()
    SANDSTORM = auto()
    DUSTSTORM = auto()
    POLLINATE = auto()
    SWARMSIGNAL = auto()
    SMOGSPREAD = auto()
    SPRINKLE = auto()
    AURAPROJECTION = auto()
    HAUNT = auto()
    DAYDREAM = auto()
    DRAGONFORCE = auto()
    SUPERCELL = auto()
    MAGNETIZE = auto()
    STRONGWINDS = auto()
    CATACLYSMICLIGHT = auto()

    DESOLATELAND = auto()
    PRIMORDIALSEA = auto()
    DELTASTREAM = auto()

    CLIMATE_WEATHERS: ClassVar[FrozenSet["Weather"]]
    IRRITANT_WEATHERS: ClassVar[FrozenSet["Weather"]]
    ENERGY_WEATHERS: ClassVar[FrozenSet["Weather"]]
    CLEARING_WEATHERS: ClassVar[FrozenSet["Weather"]]
    CATACLYSM_WEATHERS: ClassVar[FrozenSet["Weather"]]

    def is_climate_weather(self) -> bool:
        return self in Weather.CLIMATE_WEATHERS

    def is_irritant_weather(self) -> bool:
        return self in Weather.IRRITANT_WEATHERS

    def is_energy_weather(self) -> bool:
        return self in Weather.ENERGY_WEATHERS

    def is_clearing_weather(self) -> bool:
        return self in Weather.CLEARING_WEATHERS

    def is_cataclysm_weather(self) -> bool:
        return self in Weather.CATACLYSM_WEATHERS

    def __str__(self) -> str:
        return f"{self.name} (weather) object"

    @staticmethod
    def from_showdown_message(message: str, warn: bool = True):
        """Returns the Weather object corresponding to the message.

        :param message: The message to convert.
        :type message: str
        :param warn: Whether to warn about unknown weather strings.
        :type warn: bool
        :return: The corresponding Weather object.
        :rtype: Weather
        """
        message = message.replace("ability: ", "")
        message = message.replace("move: ", "")
        message = message.replace(" ", "_")
        message = message.replace("-", "_")

        try:
            return Weather[message.upper()]
        except KeyError:
            compact_message = message.replace("_", "")
            if compact_message.upper() in Weather.__members__:
                return Weather[compact_message.upper()]

            if warn:
                logging.getLogger("poke-env").warning(
                    "Unexpected weather '%s' received. Weather.UNKNOWN will be used "
                    "instead. If this is unexpected, please open an issue at "
                    "https://github.com/hsahovic/poke-env/issues/ along with this error "
                    "message and a description of your program.",
                    message,
                )
            return Weather.UNKNOWN

Weather.CLIMATE_WEATHERS = frozenset(
    {
        Weather.SUNNYDAY,
        Weather.DESOLATELAND,
        Weather.RAINDANCE,
        Weather.PRIMORDIALSEA,
        Weather.HAIL,
        Weather.SNOWSCAPE,
        Weather.BLOODMOON,
        Weather.FOGHORN,
    }
)
Weather.IRRITANT_WEATHERS = frozenset(
    {
        Weather.SANDSTORM,
        Weather.DUSTSTORM,
        Weather.POLLINATE,
        Weather.SWARMSIGNAL,
        Weather.SMOGSPREAD,
        Weather.SPRINKLE,
    }
)
Weather.ENERGY_WEATHERS = frozenset(
    {
        Weather.AURAPROJECTION,
        Weather.HAUNT,
        Weather.DAYDREAM,
        Weather.DRAGONFORCE,
        Weather.SUPERCELL,
        Weather.MAGNETIZE,
    }
)
Weather.CLEARING_WEATHERS = frozenset({Weather.STRONGWINDS, Weather.DELTASTREAM})
Weather.CATACLYSM_WEATHERS = frozenset({Weather.CATACLYSMICLIGHT})
