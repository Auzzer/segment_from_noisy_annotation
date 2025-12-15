#  Prediction
Note that we use the given data structure to deploy it:
-data
-- images
-- labels

You can access our conda environment and exccute code with the following steps directly:

`conda activate /projectnb/ec500kb/projects/Fall_2025_Projects/vessel_seg/noisy_seg_env`

then excute python code like:

python -m model_script.predict.run_mtcl_with_lungmask \
  --data_dir /projectnb/ec500kb/projects/Fall_2025_Projects/vessel_seg/data_testing \
  --output_dir ./temp/results/mtcl_lungmasked
# Results
1. you can check the model weights in `/projectnb/ec500kb/projects/Fall_2025_Projects/vessel_seg/results`

2. the predicted results are saved to:

10unets and ensemble: `/projectnb/ec500kb/projects/Fall_2025_Projects/vessel_seg/data/unet_prediction`

and mtcl: `/projectnb/ec500kb/projects/Fall_2025_Projects/vessel_seg/data/mtcl`

3. the frangi bo results are: `/projectnb/ec500kb/projects/Fall_2025_Projects/vessel_seg/data/frangi_lungmasked`

They are obtained from scripts:
`/projectnb/ec500kb/projects/Fall_2025_Projects/vessel_seg/model_script/bo`


the training history is in: `/projectnb/ec500kb/projects/Fall_2025_Projects/vessel_seg/results/output_frangi_bo_seed0`


# docker
Alternatively, you can use docker:

## 1. switch to the project directory
cd /projectnb/ec500kb/projects/Fall_2025_Projects/vessel_seg

## 2. maybe monai is requred to be installed again:
singularity exec --nv my-app.sif \
  python -m pip install --no-deps --target /projectnb/ec500kb/projects/Fall_2025_Projects/vessel_seg/singularity_pkgs \
  monai==1.5.1

## 3. Then run prediction:
SINGULARITYENV_PYTHONPATH="/projectnb/ec500kb/projects/Fall_2025_Projects/vessel_seg/singularity_pkgs:$PYTHONPATH" \
singularity run --nv my-app.sif \
  --data_dir /projectnb/ec500kb/projects/Fall_2025_Projects/vessel_seg/data_testing \
  --output_dir /projectnb/ec500kb/projects/Fall_2025_Projects/vessel_seg/data/mtcl_lungmasked

** Again, we use the given data structure. So make sure your data is organized as:
-data_testing changed to your folder name
-- images
-- labels
