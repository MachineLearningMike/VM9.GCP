gcloud compute scp --recurse vm7:/mnt/data/Trading/Checkpoints/v4.05.300.11.70.eth.30.12* /mnt/data/Trading/Checkpoints --zone asia-southeast1-c
gcloud compute scp --recurse vm7:/mnt/data/Trading/CSVLogs/v4.05.300.11.70.eth.30.12* /mnt/data/Trading/CSVLogs --zone asia-southeast1-c
# gcloud compute scp --recurse vm7:/mnt/data/Trading/Candles/table-23-06* /mnt/data/Trading/Candles --zone asia-southeast1-c

gcloud compute scp --recurse vm9:/mnt/data/Trading/Checkpoints/v4.05.300.11.70.eth.50.12* /mnt/data/Trading/Checkpoints --zone asia-southeast1-c
gcloud compute scp --recurse vm9:/mnt/data/Trading/CSVLogs/v4.05.300.11.70.eth.50.12* /mnt/data/Trading/CSVLogs --zone asia-southeast1-c
# gcloud compute scp --recurse vm7:/mnt/data/Trading/Candles/table-23-06* /mnt/data/Trading/Candles --zone asia-southeast1-c