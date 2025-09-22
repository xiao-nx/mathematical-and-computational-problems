import numpy as np
from collections import Counter
from math import comb, factorial
import matplotlib.pyplot as plt

def count_colorings_without_rotation(num_faces, num_colors, max_usage):
    """
    Calculates the number of valid colorings without considering rotational symmetry.
    
    This version correctly finds all unique partitions and calculates the
    combinatorics for each.
    
    Parameters:
    - num_faces (int): The number of faces to color (e.g., 6 for a cube).
    - num_colors (int): The total number of available colors.
    - max_usage (int): The maximum number of times a single color can be used.
    
    Returns:
    - int: The total number of valid colorings.
    """
    
    total_ways = 0
    memo = {}

    def find_unique_patterns(target_sum, num_parts, max_val, min_val=0):
        """
        Recursively finds all unique integer partitions of target_sum into num_parts,
        with each part between min_val and max_val.
        Returns a list of unique, sorted partitions.
        """
        state = (target_sum, num_parts, max_val, min_val)
        if state in memo:
            return memo[state]
            
        if num_parts == 1:
            if min_val <= target_sum <= max_val:
                return [[target_sum]]
            else:
                return []
        
        patterns = []
        # Iterate in descending order to find unique partitions
        for i in range(min(target_sum, max_val), min_val - 1, -1):
            sub_patterns = find_unique_patterns(target_sum - i, num_parts - 1, i, min_val)
            for sub_pattern in sub_patterns:
                patterns.append([i] + sub_pattern)
        
        memo[state] = patterns
        return patterns

    # Find all unique partitions of num_faces into num_colors
    # This generates patterns like [2, 2, 1, 1, 0] directly.
    usage_patterns = find_unique_patterns(num_faces, num_colors, max_usage)

    # For each unique pattern, calculate the total number of ways
    for pattern in usage_patterns:
        # Step 1: Count how many ways to choose colors for this pattern
        # The pattern is like [2, 2, 1, 1, 0]
        # We need to group identical counts.
        pattern_counts = Counter(pattern)
        
        # Example: {2: 2, 1: 2, 0: 1}
        # Choose 2 colors for usage count 2: C(5,2)
        # Choose 2 colors for usage count 1: C(3,2)
        # Choose 1 color for usage count 0: C(1,1)
        # The number of ways to choose colors is the multinomial coefficient
        # num_colors! / (freq1! * freq2! * ...)
        ways_to_choose_colors = factorial(num_colors) / np.prod([factorial(freq) for freq in pattern_counts.values()])

        # Step 2: Count how many ways to arrange faces for this pattern
        # This is the multinomial coefficient for assigning faces to the colors
        ways_to_assign_faces = factorial(num_faces) / np.prod([factorial(count) for count in pattern])
        
        # Total ways for this pattern
        total_ways += ways_to_choose_colors * ways_to_assign_faces
        
    return int(total_ways)


if __name__ == "__main__":
    c = 5  # Number of colors
    m = 2  # Max usage per color
    
    total_colorings = count_colorings_without_rotation(num_faces=6, num_colors=c, max_usage=m)
    
    print(f"Number of colors (c): {c}")
    print(f"Max usage per color (m): {m}")
    print(f"Number of total colorings (without rotation): {total_colorings}")

    # You can also manually verify the calculation:
    # Pattern [2, 2, 2, 0, 0]: comb(5,3) * factorial(6)/(2!2!2!) = 10 * 90 = 900
    # Pattern [2, 2, 1, 1, 0]: comb(5,2)*comb(3,2) * factorial(6)/(2!2!1!1!) = 10 * 3 * 180 = 5400
    # Pattern [2, 1, 1, 1, 1]: comb(5,1)*comb(4,4) * factorial(6)/(2!1!1!1!1!) = 5 * 1 * 360 = 1800
    # Total = 900 + 5400 + 1800 = 8100.
    
    
