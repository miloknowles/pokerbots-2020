# This script will be run from the top level directory of the bot.
rm -rf build/ && mkdir build && cd build

# For pure C++ bot turn these off.
cmake .. -DBUILD_PYTHON_WRAPPER=OFF -DBUILD_TESTS=OFF
make -j8

cd ..
