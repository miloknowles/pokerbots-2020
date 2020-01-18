# This script will be run from the top level directory of the bot.
PREV_BUILD_USER=$(head -n 1 ./.prev_build_user)
echo "\nNOTE: Project was last built by = ${PREV_BUILD_USER}\n"

# Check if $USER has changed, if so do a clean build.
if [ "${USER}" != "${PREV_BUILD_USER}" ]
then
  echo "WARNING: current user ${USER} != ${PREV_BUILD_USER}, cleaning"
  rm -rf build/ && mkdir build
  echo "${USER}" > .prev_build_user
fi

# Do the cmake build.
cd build
cmake .. -DBUILD_PYTHON_WRAPPER=OFF -DBUILD_TESTS=OFF
make -j2

cd ..
