| Metric | What it measures | Best case | Worst case | Example |
|------|------|------|------|------|
| success_rate | Percentage of episodes where the robot reaches the goal | 1.0 (100%) | 0.0 | 0.24 → robot succeeds 24% of the time |
| violations_per_100_steps | Number of safety violations per 100 steps | 0 | high value | 58 → many collisions |
| avg_time_to_failure | Average time before the first safety violation | high value | small value | 94 → robot stays safe longer |
| mean_episode_length | Average episode duration | close to max_steps | very small | 300 → robot survives the whole episode |
| avg_collisions_per_episode | Average number of collisions in one episode | 0 | large value | 22 → robot hits obstacles often |
| avg_episode_reward | Average reward accumulated during the episode | large positive | large negative | -1079 → robot receives heavy penalties |