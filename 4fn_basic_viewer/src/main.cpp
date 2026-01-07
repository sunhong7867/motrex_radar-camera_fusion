
#include "socket.hpp"
#include "data_handler.hpp"

int main(int argc, char **argv)
{
    ros::init(argc, argv, "radar_node");
    ros::NodeHandle nh;  // Node Handler 

    ros::Rate loop(100);
    DataHandler handler(nh);
    
    // Initialize the radar packet data handler
    handler.init();

	while (ros::ok()) {
		if(!handler.receive()) {
			ROS_DEBUG("failed to receive the packet data !!!");
            continue;
		} 

        if(!handler.publish()) {
            ROS_ERROR("failed to publish the points to RVIZ");
        }

        loop.sleep();        
	}
	
	return 0;
}

