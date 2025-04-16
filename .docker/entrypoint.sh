set -e

# setup ros environment
source "/opt/ros/noetic/setup.bash"
source "/potato_ws/devel/setup.bash"

exec "$@"