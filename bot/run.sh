# source setup.sh
export LD_LIBRARY_PATH=`pwd`/pokerbots_cpp/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=`pwd`/build/pokerbots_cpp/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/lib:/usr/lib/x86_64-linux-gnu:/usr/lib:$LD_LIBRARY_PATH

echo $LD_LIBRARY_PATH

cd build
chmod +x cpp_player
./cpp_player --port $1
