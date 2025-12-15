#  Prediction
You can access our conda environment and exccute code with the following steps directly:

`conda activate /projectnb/ec500kb/projects/Fall_2025_Projects/vessel_seg/noisy_seg_env`

then excute python code like:

python -m model_script.predict.run_mtcl_with_lungmask \
  --data_dir /projectnb/ec500kb/projects/Fall_2025_Projects/vessel_seg/data \
  --output_dir ./temp/results/mtcl_lungmasked
# Results
1. you can check the model weights in `/projectnb/ec500kb/projects/Fall_2025_Projects/vessel_seg/results`

2. the predicted results are saved to:

10unets and ensemble: `/projectnb/ec500kb/projects/Fall_2025_Projects/vessel_seg/data/unet_prediction`

and mtcl: `/projectnb/ec500kb/projects/Fall_2025_Projects/vessel_seg/data/mtcl`

3. the frangi bo results are: `/projectnb/ec500kb/projects/Fall_2025_Projects/vessel_seg/data/frangi_lungmasked`

They are obtained from scripts:
`/projectnb/ec500kb/projects/Fall_2025_Projects/vessel_seg/model_script/bo`


the training his is in: `/projectnb/ec500kb/projects/Fall_2025_Projects/vessel_seg/results/output_frangi_bo_seed0`
