# DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
# DIR=`dirname $0`
# echo "The directory is ${DIR}"

# rm -rf ./pokerbots_cpp_python/build && mkdir ./pokerbots_cpp_python/build
# cd ./pokerbots_cpp_python/build && cmake .. && make -j8

# Return to original directory.
# cd $DIR
# cd ../../

rm -rf build/ && mkdir build && cd build
cmake .. && make -j8
