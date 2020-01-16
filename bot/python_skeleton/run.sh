# export LD_LIBRARY_PATH=`pwd`:$LD_LIBRARY_PATH
# echo $LD_LIBRARY_PATH
# python3.5 player.py
cd build && ./cpp_player --port $1
