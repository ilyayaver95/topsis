

def knapsack(items, max_weight):
    # Create a 2D list to store the subproblems
    dp = [[0 for _ in range(max_weight + 1)] for _ in range(len(items) + 1)]
    # Iterate through all items
    for i in range(1, len(items) + 1):
        item, weight, value = items[i - 1]
        # Iterate through all possible weights
        for w in range(max_weight + 1):
            # Case 1: Don't include the item
            dp[i][w] = dp[i - 1][w]
            # Case 2: Include the item
            if weight <= w:
                dp[i][w] = max(dp[i][w], dp[i - 1][w - weight] + value)
    # The maximum value is in the bottom right corner
    return dp[-1][-1]



# Test the function
items = [("item1", 20, 30), ("item2", 18, 25), ("item3", 17, 20), ("item4", 15, 18), ("item4", 15, 17)
         , ("item4", 10, 11), ("item4", 5, 5), ("item4", 3, 2), ("item4", 1, 1), ("item4", 1, 1)]  # , ("item5", 9, 10)

print(f"The maximum value of the sack is {knapsack(items, 70)}")
