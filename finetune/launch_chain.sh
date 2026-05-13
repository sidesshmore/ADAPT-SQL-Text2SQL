#!/bin/bash
# Submit 3 chained GRPO jobs. Each backup only runs if the previous job failed.
# Usage: cd /scratch/vavaghad/sidessh/ADAPT-SQL-Text2SQL && bash finetune/launch_chain.sh

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
FINETUNE_SH=$SCRIPT_DIR/finetune_gaudi.sh

JOB1=$(sbatch --parsable "$FINETUNE_SH")
echo "Job 1 (primary):  $JOB1"

JOB2=$(sbatch --parsable --dependency=afternotok:$JOB1 "$FINETUNE_SH")
echo "Job 2 (backup 1): $JOB2  [starts only if $JOB1 fails]"

JOB3=$(sbatch --parsable --dependency=afternotok:$JOB2 "$FINETUNE_SH")
echo "Job 3 (backup 2): $JOB3  [starts only if $JOB2 fails]"

echo ""
echo "Monitor:    squeue -u \$USER -j $JOB1,$JOB2,$JOB3"
echo "Cancel all: scancel $JOB1 $JOB2 $JOB3"
