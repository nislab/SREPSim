"""
Minimal functional testing of the simulator.

Author: Novak Boškov <boskov@bu.edu>
Date: September, 2022.
"""

from typing import Dict, Any, Optional

from srep_simulator.srep import Stats


def stat_compare(s: Stats, to_compare: Dict[str, Any]) -> Optional[bool]:
    """Compare `s` attributes using only those in `to_compare`."""
    for k, v in to_compare.items():
        if not getattr(s, k) == v:
            raise ValueError(f"{k} in {s} not equal to {v}")

    return True
