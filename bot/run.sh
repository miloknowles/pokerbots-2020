# source setup.sh
export LD_LIBRARY_PATH=`pwd`/build/:`pwd`/build/pokerbots_cpp/:`pwd`/pokerbots_cpp/:$LD_LIBRARY_PATH
echo $LD_LIBRARY_PATH
cd build
chmod +x cpp_player
./cpp_player --port $1
