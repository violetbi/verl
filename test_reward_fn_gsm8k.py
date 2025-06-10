from typing import Dict, Any, Callable
from verl.trainer.main_ppo import get_custom_reward_fn

#from verl.trainer.ppo.reward import get_custom_reward_fn

def load_reward_function(config: Dict[str, Any]) -> Callable[[str, str], float]:
    """Load and validate a custom reward function from configuration.
    
    Args:
        config: Configuration dictionary containing:
            - custom_reward_function: Dictionary with:
                - path: File path to the reward module
                - name: Function name to load
    
    Returns:
        A callable reward function that takes (solution_str, ground_truth) and returns a float
    
    Raises:
        AssertionError: If the reward function fails to load
    """
    reward_fn = get_custom_reward_fn(config)
    #assert reward_fn is not None, "Failed to load reward function from config"
    return reward_fn

def evaluate_solution(
    reward_fn: Callable[[str, str], float],
    solution_str: str,
    ground_truth: str,
    verbose: bool = True
) -> float:
    """Evaluate a solution against ground truth using the provided reward function.
    
    Args:
        reward_fn: Function that computes reward score
        solution_str: Proposed solution string to evaluate
        ground_truth: Correct reference answer
        verbose: Whether to print the computed reward
    
    Returns:
        The computed reward score as float
    """
    reward = reward_fn(solution_str, ground_truth)
    if verbose:
        print(f"Solution: {solution_str[:50]}... | Ground Truth: {ground_truth} | Reward: {reward:.2f}")
    return reward

def run_test_cases() -> None:
    """Run test cases for reward function evaluation."""
    # Configuration for reward function
    #config ={"custom_reward_function": "invalid_value"}
    
    config = {
        "custom_reward_function": {
            #"path": "",
            "path": "./verl/utils/reward_score/gsm8k.py",
            "name": "compute_score"
        }
    }
    
    # Load reward function
    reward_fn = load_reward_function(config)
    


    test_cases = [
        {
            "name": "correct_match",
            "solution_str": """Natalia sold 48/2 = <<48/2=24>>24 clips in May.\n"
                   "Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n"
                   "#### 72""",
            "ground_truth": "72",
            "expected": 1.0  # Assuming correct match gives full reward
        },
        {
            "name": "incorrect_match",
            "solution_str": """ Natalia sold 48/2 = <<48/2=24>>24 clips in May.\n"
                   "Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n"
                   "#### 24""",
            "ground_truth": "72",
            "expected": 0.0  # Assuming incorrect match gives no reward
        }
    ]
    
    for case in test_cases:
        print(f"\nRunning test case: {case['name']}")
        reward = evaluate_solution(
            reward_fn,
            case["solution_str"],
            case["ground_truth"]
        )
        
        # Basic verification (adjust expected values as needed)
        if case["expected"] is not None:
            assert reward == case["expected"], (
                f"Test failed: {case['name']}\n"
                f"Expected: {case['expected']}, Got: {reward}"
            )

if __name__ == "__main__":
    run_test_cases()