PYTHONNOUSERSITE=1 scripts/vipe_to_scene.py \
    --artifacts_dir vipe_results/cosmos-example \
    --artifact_name cosmos-example \
    --output_dir vipe_results/cosmos-example/scene

PYTHONNOUSERSITE=1 python agentic-cinematography/scene-processing/run_scene_pipeline.py \
    --scene_dir vipe_results/cosmos-example/scene
