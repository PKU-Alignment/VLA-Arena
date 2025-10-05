import os
import json
import argparse
from vla_arena.evaluation.evaluator import Evaluator
from vla_arena.evaluation.policy import SmolVLA
from vla_arena.evaluation.policy import OpenVLA
from vla_arena.evaluation.policy import OpenVLAOFT
from vla_arena.evaluation.policy import OpenPI
from vla_arena.evaluation.policy import RandomPolicy
from vla_arena.evaluation.policy import PolicyRegistry

# import debugpy
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9501))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass

os.environ["MUJOCO_GL"] = "egl"


def parse_levels(levels_str):
    """
    Parse level string to support various formats:
    - Single level: "0" -> [0]
    - Range: "0-2" -> [0, 1, 2]
    - List: "0,2" -> [0, 2]
    """
    if levels_str is None:
        return None
    
    levels = []
    parts = levels_str.split(',')
    
    for part in parts:
        part = part.strip()
        if '-' in part:
            # Handle range
            start, end = part.split('-')
            start, end = int(start.strip()), int(end.strip())
            levels.extend(list(range(start, end + 1)))
        else:
            # Handle single level
            levels.append(int(part))
    
    # Remove duplicates and sort
    levels = sorted(list(set(levels)))
    return levels


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_suite', default=None, type=str, 
                       choices=[
                           "safety_dynamic_obstacles",
                           "safety_hazard_avoidance",
                           "safety_object_state_preservation",
                           "safety_risk_aware_grasping",
                           "safety_static_obstacles",
                           "robustness_dynamic_distractors",
                           "robustness_static_distractors",
                           "robustness_visual_variations",
                           "generalization_language_variations",
                           "generalization_object_preposition_combinations",
                           "generalization_task_workflows",
                           "generalization_unseen_objects",
                           "long_horizon",
                           "libero_10",
                           "libero_90",
                           "libero_spatial",
                           "libero_object",
                           "libero_goal",
                       ], 
                       help="The evaluation track to run")
    
    # Modified: Support both single level and multiple levels
    parser.add_argument('--task_level', default="0", type=str,
                       help="Task level(s) to evaluate. Supports: single (0), range (0-2), list (0,2,4), or mixed (0-2,5)")
    
    parser.add_argument('--n-episode', default=1, type=int, 
                       help="The number of episodes to evaluate for each task")
    parser.add_argument('--policy', default="openvla", type=str,
                       choices=PolicyRegistry.list_policies(),
                       help="The policy to evaluate")
    parser.add_argument('--model_ckpt', default=None,
                       help="The base model checkpoint path")
    parser.add_argument('--save-dir', default="logs", 
                       help="The directory to save the evaluation results")
    parser.add_argument('--visualization', action="store_true", default=False, 
                       help="Whether to visualize the episodes")
    parser.add_argument('--metrics', nargs='+', 
                       default=["success_rate", "cumulative_cost", "safe_success_rate"], 
                       choices=["success_rate", "cumulative_cost", 'safe_success_rate', "episode_length"], 
                       help="The metrics to evaluate")
    parser.add_argument('--host', default="localhost", type=str, 
                       help="The host to the remote server")
    parser.add_argument('--port', default=5555, type=int, 
                       help="The port to the remote server")
    parser.add_argument('--replanstep', default=4, type=int, 
                       help="The step to replan")
    
    # Additional arguments for batch evaluation
    parser.add_argument('--parallel', action="store_true", default=False,
                       help="Whether to run level evaluations in parallel (experimental)")
    parser.add_argument('--episode_config', default=None, type=str,
                       help="Path to episode configuration file")
    
    args = parser.parse_args()
    return args


def print_evaluation_plan(args, task_levels):
    """Print the evaluation plan before starting"""
    print("\n" + "="*70)
    print("EVALUATION PLAN")
    print("="*70)
    print(f"Task Suite: {args.task_suite}")
    print(f"Levels to evaluate: {task_levels}")
    print(f"Episodes per task: {args.n_episode}")
    print(f"Policy: {args.policy}")
    print(f"Metrics: {args.metrics}")
    print(f"Visualization: {args.visualization}")
    print(f"Save directory: {args.save_dir}")
    print("="*70 + "\n")
    
    # Calculate total evaluation scope
    num_levels = len(task_levels)
    # This is approximate - actual number depends on the suite
    estimated_tasks_per_level = 10  # You might want to get this from the suite
    total_episodes = num_levels * estimated_tasks_per_level * args.n_episode
    
    print(f"Estimated total episodes: ~{total_episodes}")
    print(f"Press Ctrl+C to cancel, or wait to continue...\n")
    
    import time
    time.sleep(3)  # Give user time to cancel if needed


def evaluate(args):
    """Main evaluation function with multi-level support"""
    
    # Parse task levels
    task_levels = parse_levels(args.task_level)
    if not task_levels:
        raise ValueError("No valid task levels specified!")
    
    # Load episode configuration if provided
    episode_config = None
    if args.episode_config:
        with open(args.episode_config, 'r') as f:
            episode_config = json.load(f)
    
    # Set up save directory
    if args.task_suite is not None:
        args.save_dir = os.path.join(args.save_dir, args.task_suite)
    
    if not args.task_suite:
        raise ValueError("No tasks specified! Please provide --task_suite")
    
    # Print evaluation plan
    print_evaluation_plan(args, task_levels)
    
    print(f"Tasks to evaluate: {args.task_suite}")
    print(f"Levels to evaluate: {task_levels}")
    print(f"Number of episodes per task: {args.n_episode}")
    
    # Create evaluator with multiple levels support
    evaluator = Evaluator(
        task_suite=args.task_suite,
        task_levels=task_levels,  # Pass list of levels
        n_episodes=args.n_episode,
        episode_config=episode_config,
        max_substeps=1,  # repeat step in simulation
        save_dir=args.save_dir,
        visualization=args.visualization,
        metrics=args.metrics
    )
    if args.policy not in PolicyRegistry.list_policies():
        raise ValueError(f"Policy '{args.policy}' is not registered. Available policies: {PolicyRegistry.list_policies()}")
    if args.policy != "openpi":
        policy = PolicyRegistry.get(args.policy, model_ckpt=args.model_ckpt if args.model_ckpt else None)
    else:
        policy = PolicyRegistry.get(args.policy, host=args.host, port=args.port)
    
    # Run evaluation
    results = evaluator.evaluate(policy)
    
    # Print quick summary of results
    print("\n" + "="*70)
    print("EVALUATION COMPLETED SUCCESSFULLY")
    print("="*70)
    
    if isinstance(results, dict):
        # If single level, results is a dict of task metrics
        if len(task_levels) == 1:
            print(f"\nLevel {task_levels[0]} Results:")
            for task_name, metrics in results.items():
                print(f"  {task_name}:")
                if "success_rate" in metrics:
                    print(f"    Success Rate: {metrics['success_rate']:.2%}")
                if "safe_success_rate" in metrics:
                    print(f"    Safe Success Rate: {metrics['safe_success_rate']:.2%}")
                if "cumulative_cost" in metrics:
                    print(f"    Avg Cost: {metrics['cumulative_cost']:.2f}")
        else:
            # Multiple levels, results is dict of level -> task metrics
            for level, level_results in results.items():
                print(f"\nLevel {level} Results:")
                success_rates = []
                for task_name, metrics in level_results.items():
                    if "success_rate" in metrics:
                        success_rates.append(metrics['success_rate'])
                if success_rates:
                    avg_success = sum(success_rates) / len(success_rates)
                    print(f"  Average Success Rate: {avg_success:.2%}")
    
    print(f"\nDetailed results saved to: {evaluator.save_dir}")
        
    # except KeyboardInterrupt:
    #     print("\n\nEvaluation interrupted by user.")
    #     print("Partial results may have been saved.")
    # except Exception as e:
    #     print(f"\n\nEvaluation failed with error: {e}")
    #     import traceback
    #     traceback.print_exc()
    #     raise


def main():
    """Entry point with better error handling"""
    args = get_args()
    
    # Validate arguments
    if not args.task_suite:
        print("Error: --task_suite is required!")
        print("Available options: static_obstacles, preposition_generalization")
        return 1
    
    try:
        evaluate(args)
        return 0
    except Exception as e:
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())