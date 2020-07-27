#!/bin/bash
# submit looped chain of jobs
# run as . emulator_predictions_batch.bash
# each is a 1 hour job for 5,000 custom inputs

current=$(qsub emulator_predictions.bash)
echo $current

for id in {2..3}; do 
  current_id=$(echo $current | tr -d -c 0-9)
  next=$(qsub -hold_jid $current_id emulator_predictions.bash)
  echo $next
  current=$next;
done
