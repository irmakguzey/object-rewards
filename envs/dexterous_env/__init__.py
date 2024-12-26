from gymnasium.envs.registration import register

register(
    id="H2R-DexterousTracking-v1",
    entry_point="dexterous_env.dexterous_arm_env_with_tracking:DexterityEnvTracking",
    max_episode_steps=120,
)

register(
    id="H2R-DexterousTrackingLargerGen-v1",
    entry_point="dexterous_env.dexterous_arm_env_larger_generalization:DexterityEnvLargerGeneralization",
    max_episode_steps=120,
)
