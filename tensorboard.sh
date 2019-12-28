# Runs a tensorboard server to view training progress.
# NOTE(milo): Had to add export PATH="$PATH:/home/milo/.local/bin" to get this executable.

tensorboard --logdir /home/milo/pokerbots-2020/training_logs/$1 --port 6006 --bind_all
