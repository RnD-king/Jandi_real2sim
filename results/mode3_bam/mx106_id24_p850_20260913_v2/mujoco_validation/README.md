cd /home/noh/Jandi_real2sim

for condition in \
  mass1_distance1 mass1_distance2 \
  mass2_distance1 mass2_distance2 \
  mass3_distance1 mass3_distance2
do
  uv run jandi-r2s-mode3-bam-mujoco \
    --condition "$condition" \
    --compare-repeat 3 \
    --model both
done
