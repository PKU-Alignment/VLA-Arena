import time
import numpy as np
from vla_arena.vla_arena.envs.env_wrapper import OffScreenRenderEnv

def benchmark_reset_vs_load(bddl_file, num_episodes=10):
    print(f"🚀 Benchmark: Reset (Random) vs. Load (Cached) for: {bddl_file.split('/')[-1]}")
    print("-" * 60)

    # --- 阶段 1: 环境构建 (Init) ---
    print("Step 1: Constructing Environment...")
    t_start_init = time.perf_counter()
    
    env_args = {
        'bddl_file_name': bddl_file,
        'camera_heights': 256,
        'camera_widths': 256,
    }
    env = OffScreenRenderEnv(**env_args)
    t_end_init = time.perf_counter()
    init_duration = t_end_init - t_start_init
    print(f"✅ Env Init finished in: {init_duration:.4f}s\n")

    # --- 阶段 2: Reset (随机生成) ---
    print(f"Step 2: Benchmarking 'env.reset()' ({num_episodes} times)...")
    reset_times = []
    
    # 我们保留最后一次的状态作为“Cached State”给下一阶段用
    cached_state = None 

    for i in range(num_episodes):
        t_start = time.perf_counter()
        
        # 核心操作：随机重置
        env.reset()
        env.seed(42)  # 固定随机种子，确保可复现

        
        t_end = time.perf_counter()
        reset_times.append(t_end - t_start)

        # 第一次通常较慢 (Cold Start)
        tag = "[COLD]" if i == 0 else "[WARM]"
        print(f"  Reset {i+1:02d} {tag}: {reset_times[-1]:.4f}s")

    # 获取一个合法的状态用于测试
    cached_state = env.get_sim_state()
    print(f"✅ Average Warm Reset Time: {np.mean(reset_times[1:]):.4f}s\n")

    # --- 阶段 3: Load Cached State (直接加载) ---
    print(f"Step 3: Benchmarking 'env.set_init_state()' ({num_episodes} times)...")
    load_times = []

    # 注意：这里我们假设 env.set_init_state 存在 (VLA-Arena/Libero 标准接口)
    # 如果你的 wrapper 叫 set_state 或其他名字，请在此调整
    if not hasattr(env, 'set_init_state'):
        print("⚠️ Warning: env.set_init_state not found. Trying fallback methods...")
        # 简单的 fallback 逻辑用于演示
        if hasattr(env, 'set_sim_state'): 
            def set_init_state_shim(s): env.set_sim_state(s); return env.get_observation()
            env.set_init_state = set_init_state_shim
        else:
            raise AttributeError("Cannot find set_init_state method to benchmark.")

    for i in range(num_episodes):
        t_start = time.perf_counter()
        
        # 核心操作：加载缓存状态 + 获取 Observation
        # 这里的关键是：它跳过了物理 settle 和随机化过程
        obs = env.set_init_state(cached_state)
        
        t_end = time.perf_counter()
        load_times.append(t_end - t_start)
        
        print(f"  Load  {i+1:02d}       : {load_times[-1]:.4f}s")

    # --- 阶段 4: 最终对比报告 ---
    env.close()
    
    avg_reset = np.mean(reset_times[1:]) # 排除冷启动
    avg_load = np.mean(load_times)
    speedup = avg_reset / avg_load

    print("\n" + "="*60)
    print("🏆 PERFORMANCE SHOWDOWN")
    print("="*60)
    print(f"1. Init Environment : {init_duration:.4f}s (One-time cost)")
    print("-" * 60)
    print(f"2. Random Reset     : {avg_reset:.4f}s / episode")
    print(f"   (Includes: Randomization + Physics Settle + Rendering)")
    print("-" * 60)
    print(f"3. Load Cached State: {avg_load:.4f}s / episode")
    print(f"   (Includes: Memory Copy + Forward + Rendering)")
    print("="*60)
    print(f"🚀 CONCLUSION: Loading cached states is {speedup:.1f}x FASTER than resetting!")
    print("="*60)

# 使用示例
# bddl_path = "path/to/your/file.bddl"
# benchmark_reset_vs_load(bddl_path)
if __name__ == "__main__":
    bddl_file = "/home/zhangborong/VLA-Arena-pub/vla_arena/vla_arena/bddl_files/distractor_dynamic_distractors/level_0/pick_up_the_banana_and_put_it_on_the_plate.bddl"
    benchmark_reset_vs_load(bddl_file, num_episodes=10)