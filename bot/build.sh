# This script will be run from the top level directory of the bot.
# PREV_BUILD_DIR=$(head -n 1 .PREV_BUILD_DIR)

# if [ "${PREV_BUILD_DIR}" != `pwd` ]
# then
#   echo "WARNING: current top level directory `pwd` != ${PREV_BUILD_DIR}, cleaning"
#   rm -rf build/ && mkdir build
#   echo `pwd` > .PREV_BUILD_DIR
#   echo "Updated PREV_BUILD_DIR to `pwd`"
#   cat .PREV_BUILD_DIR
# fi

# Do the cmake build.
rm -rf build/
mkdir build
cd build

# Turn off everything we don't need to build.
cmake .. -DBUILD_PYTHON_WRAPPER=OFF -DBUILD_TESTS=OFF -DBUILD_CFR_EXECUTABLES=OFF -DBUILD_CLUSTERING=OFF
make

cd ..
