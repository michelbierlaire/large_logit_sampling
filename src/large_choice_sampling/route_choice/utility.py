"""Utility model for route-based alternatives.

This module defines a simple utility specification for route choice models.
The utility is assumed to depend linearly on total travel time, with a
negative sign to reflect disutility of travel time.

Michel Bierlaire
Wed Mar 25 2026, 14:52:33
"""

from __future__ import annotations

from dataclasses import dataclass

from .alternative import RouteAlternative


@dataclass(frozen=True)
class RouteUtilityModel:
    """Utility model for route alternatives.

    This model defines the systematic utility of a route as a linear function
    of its total travel time.

    :param beta_time: Marginal utility of travel time (typically negative in
        behavioral interpretation, but implemented here with a minus sign in
        the utility function).
    """

    beta_time: float = 1.0

    def utility(self, alternative: RouteAlternative) -> float:
        """Compute the utility of a route alternative.

        The utility is defined as the negative of travel time weighted by the
        parameter ``beta_time``.

        :param alternative: Route alternative for which the utility is
            evaluated.
        :return: Utility value associated with the alternative.
        """
        return -self.beta_time * alternative.total_travel_time
