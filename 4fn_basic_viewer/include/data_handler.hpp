#ifndef DATA_HANDLER_H_
#define DATA_HANDLER_H_

#include <string>
#include <array>
#include "ros/ros.h"
#include <sensor_msgs/PointCloud2.h>
#include "socket.hpp"

// Class
class DataHandler
{
    public : 
        DataHandler(ros::NodeHandle &nh);
        ~DataHandler() {}
        void    init();
        int8_t  receive();
        int8_t  publish();

        const double pi = acos(-1);

    private :
        ros::NodeHandle                                     handle;
        std::array<point_data_t, MAX_NUM_POINTS_PER_FRAME>  r_data;

        // Publisher
        ros::Publisher                                      pub_points;

        // Messages to be published
        sensor_msgs::PointCloud2                            radarPoint;

        uint32_t                                            frame_number, nPoints;
        Socket                                              m_socket;
        int                                                 z_offset;

        float   polar2x(float range, float azimuth, float elevation);
        float   polar2y(float range, float azimuth, float elevation);
        float   polar2z(float range, float azimuth, float elevation);
};
#endif  // DATA_HANDLER_H_