source setup.sh
export LD_LIBRARY_PATH=`pwd`/pokerbots_cpp/:$LD_LIBRARY_PATH
echo $LD_LIBRARY_PATH
cd build && ./cpp_player --port $1
