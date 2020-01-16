export LD_LIBRARY_PATH=`pwd`/pokerbots_cpp_python/:$LD_LIBRARY_PATH
echo $LD_LIBRARY_PATH
# python3.5 player.py
cd build && ./cpp_player --port $1
