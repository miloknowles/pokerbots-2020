# gcloud compute ssh --project robustroboticsgroup-225320 \
#   --zone us-east1-d milo-pytorch-16cpu-32mem-t4-vm \
#   -- -L 8080:localhost:8080

gcloud compute ssh --project robustroboticsgroup-225320 \
  --zone us-east1-b milo-pytorch-cpu-only-vm \
  -- -L 8080:localhost:8080
