"""
Terminal color codes for formatted output.

This module provides a tcolors class with ANSI color codes for terminal output.
Use these constants to format print statements with colors and styles.

Author: Chiara Bissolotti (cbissolotti@anl.gov)
"""


class tcolors:
    """
    Terminal color codes for formatted output.

    Usage:
        from utilities.colors import tcolors

        print(f"{tcolors.GREEN}Success!{tcolors.ENDC}")
        print(f"{tcolors.WARNING}Warning message{tcolors.ENDC}")
        print(f"{tcolors.BOLDBLUE}Bold blue text{tcolors.ENDC}")
    """

    # Basic colors
    BLUE = "\033[94m"  # Bright blue
    CYAN = "\033[96m"  # Cyan
    GREEN = "\033[92m"  # Green
    WARNING = "\033[93m"  # Yellow
    FAIL = "\033[91m"  # Red
    GIULIA = "\033[35m"  # Purple
    OKLIGHTBLUE = "\033[34m"  # Light blue (darker blue)
    WHITE = "\033[97m"  # White

    # Text styles
    ENDC = "\033[0m"  # Reset/end color
    BOLD = "\033[1m"  # Bold text
    UNDERLINE = "\033[4m"  # Underlined text

    # Combined styles (bold + color)
    BOLDBLUE = BOLD + BLUE
    BOLDCYAN = BOLD + CYAN
    BOLDGREEN = BOLD + GREEN
    BOLDWARNING = BOLD + WARNING
    BOLDFAIL = BOLD + FAIL
    BOLDLIGHTBLUE = BOLD + OKLIGHTBLUE
    BOLDWHITE = BOLD + WHITE

    # Underlined styles
    UNDERLINEBLUE = UNDERLINE + BLUE
    UNDERLINEGREEN = UNDERLINE + GREEN
    UNDERLINEWARNING = UNDERLINE + WARNING
    UNDERLINEFAIL = UNDERLINE + FAIL
