#!/bin/bash
# Train the LUPUS walk-kick teacher (Phase 3) with the privileged ball state.
# Required: REGISTRY=file:///absolute/lupus_kick_bundle[,file:///another/bundle].
# Defaults: LOGGER=wandb, NGPUS=2 (2x A6000); LOGGER=disabled keeps logging local.
# EXP_PRESET overrides the preset, e.g. exp:lupus-kickdir-teacher or
# exp:lupus-dribble-teacher (Phase 6). Optional: WANDB_PROJECT, WANDB_ENTITY, RUN_NAME.
# Additional CLI arguments are passed through to wbt_training.train_agent,
# e.g. --training.num-envs 8192 or --reward.terms.ball-near-foot.weight 1.0.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/setup_env.sh"
php_launcher_init
NGPUS="${NGPUS:-2}"
EXP_PRESET="${EXP_PRESET:-exp:lupus-kick-teacher}"
php_positive_integer NGPUS "$NGPUS"
php_prepare_inputs teacher ''
php_training_command
php_select_training_gpu
php_activate_sim
exec "${TRAIN_COMMAND[@]}" -m wbt_training.train_agent \
    "$EXP_PRESET" simulator:isaacsim \
    --training.registry-name "$REGISTRY" \
    "${LOGGER_ARGS[@]}" --logger.video.enabled False \
    "$@"
