#!/usr/bin/env python3
import rospy
from std_msgs.msg import Float32
from stable_baselines3 import PPO
from potatoes import PotatoTowerEnv  

def publish_height():
    rospy.init_node('tower_height_publisher', anonymous=True)
    pub = rospy.Publisher('/potato_tower/height', Float32, queue_size=10)
    rate = rospy.Rate(0.2)  # 0.2 Hz = every 5 seconds

    # Load environment and model
    env = PotatoTowerEnv(render_mode=None)
    model = PPO.load("/potato_ws/src/models/ppo_potato_01.zip")

    obs, _ = env.reset()
    done = False

    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, _, _ = env.step(action)

    height = env.tower_height  # Final height after one episode
    rospy.loginfo(f" Final tower height: {height}")
    
    while not rospy.is_shutdown():
        pub.publish(height)
        rate.sleep()

if __name__ == '__main__':
    try:
        publish_height()
    except rospy.ROSInterruptException:
        pass
