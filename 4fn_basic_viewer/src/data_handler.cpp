#include "data_handler.hpp"

DataHandler::DataHandler(ros::NodeHandle &nh)
{
    handle = nh;
}

void DataHandler::init()
{
    int         numTry = 0;
	std::string ip_addr;

    // Get Radar IP address
    if (!(handle.getParam("/radar_node/ip_addr", ip_addr))) {
        ROS_ERROR("failed to get IP address");
        return; 
    } else {
        ROS_INFO("IP Address : %s", ip_addr.c_str());
    }

    // Get Port Number
    if (!(handle.getParam("/radar_node/z_offset", z_offset))) {
        ROS_ERROR("failed to get z offset");
        return; 
    } else {
        ROS_INFO("Z Offset : %d", z_offset);
    }

    while(!m_socket.connectSocket(ip_addr.c_str(), RADAR_DATA_RX_PORT)) {
        numTry++;
        if(numTry <= 20) { // 20 tries
            usleep(100000);
        } else {
            ROS_ERROR("Tried 20 times, but couldn't connect. Bye !");
            return;
        }
    }

	ROS_INFO("Connection succeeded !");

	// Point cloud publisher
	pub_points = handle.advertise<sensor_msgs::PointCloud2>("point_cloud", 1);

    // Set the basic properties of pointcloud 
    radarPoint.header.frame_id = "retina_link";
    radarPoint.height = 1;
    radarPoint.fields.resize(5);
    // Convert x/y/z to fields
    radarPoint.fields[0].name = "x"; 
    radarPoint.fields[1].name = "y"; 
    radarPoint.fields[2].name = "z";
    radarPoint.fields[3].name = "power";
    radarPoint.fields[4].name = "doppler";

    int offset = 0;
    // All offsets are *4, as all field data types are float32
    for (int d = 0; d < radarPoint.fields.size(); ++d, offset += 4) {
        radarPoint.fields[d].offset = offset;
        radarPoint.fields[d].datatype = sensor_msgs::PointField::FLOAT32;
        radarPoint.fields[d].count = 1;
    }

    radarPoint.point_step = offset;
    radarPoint.data.reserve(MAX_NUM_POINTS_PER_FRAME * radarPoint.point_step);
    radarPoint.is_bigendian = false;  
    radarPoint.is_dense = false;
}

int8_t DataHandler::receive()
{
    packet_buffer_t packetBuffer;
	int             readBytes = 0;
    static int      format_type = 0xFF;
	
    readBytes = m_socket.readData((uint8_t *)&packetBuffer.cmdHeader, RADAR_CMD_HEADER_LENGTH, true);
    if(readBytes == RADAR_CMD_HEADER_LENGTH) {
        if(memcmp(&packetBuffer.cmdHeader.header, NETWORK_TX_HEADER, NETWORK_TX_HEADER_LENGTH) != 0) {
            ROS_ERROR("Not match with the TI header magic number !!!");
            return 0;
        }

        if(packetBuffer.cmdHeader.dataSize > MAX_BUF_SIZE) {
            ROS_ERROR("Greater than max buffer size !");
            return 0;
        }
    } else {
        ROS_DEBUG("Read bytes(%d) is not matching the data size !!!", readBytes);
		return 0;
    }

    readBytes = m_socket.readData((uint8_t *)&packetBuffer.buf, FRAME_SIZE, true);
    if (readBytes == FRAME_SIZE)
    {
        if(memcmp(packetBuffer.pkHeader.magicWord, radarMagicWord, RADAR_OUTPUT_MAGIC_WORD_LENGTH) != 0)
        {
            ROS_DEBUG("Magic Word is not matched !!!");
            return 0;
        }

        // Frame Number
        frame_number = packetBuffer.pkHeader.frame_counter;
			
        // The number of points
        nPoints = packetBuffer.pkHeader.targetNumber;
		ROS_INFO("Frame Counter : %d, Total number of points : %d", frame_number, nPoints);

        // Target Format Type -> 1 : Spherical, 2 : Cartesian
        if(format_type == 0xFF) {
            format_type = packetBuffer.pkHeader.target_info_format_type;
            ROS_WARN("Format type : %d", format_type);
        }        

        for (uint32_t i = 0; i < nPoints; i++)
        {
            point_data_t    pt;
            rcv_data_t      *point = (rcv_data_t *)(packetBuffer.data + (sizeof(rcv_data_t) * i));

            if(format_type == 1) {                  // Spherical Coordinate
                float range     = point->data[0];
                float azimuth   = point->data[1];
                float elevation = point->data[2];
                pt.x            = polar2x(range, azimuth, elevation);
                pt.y            = polar2y(range, azimuth, elevation);
                pt.z            = polar2z(range, azimuth, elevation);
                pt.doppler      = point->doppler;
                pt.power        = point->power;
            } else if(format_type == 2) {           // Cartesian Coordinate
                pt.x            = point->data[0];
                pt.y            = point->data[1];
                pt.z            = point->data[2];   
                pt.doppler      = point->doppler;
                pt.power        = point->power;                
            } else {
                ROS_INFO("Not defined format type !");
                return -1;
            }  

            r_data[i] = pt;
            r_data[i].z += (z_offset/100.f);
			
			// just for debugging
            ROS_DEBUG("[%d] (%.4f, %.4f, %.4f)", i, pt.x, pt.y, pt.z);            

        }
	}

    return 1;
}

int8_t DataHandler::publish()
{
    radarPoint.header.stamp = ros::Time::now();
    radarPoint.width = nPoints;
    radarPoint.row_step = radarPoint.point_step * radarPoint.width;
    radarPoint.data.resize(nPoints * radarPoint.point_step * radarPoint.height);

    // Copy the data points
    for (int cp = 0; cp < nPoints; ++cp) {
        memcpy(&radarPoint.data[cp * radarPoint.point_step + radarPoint.fields[0].offset], &r_data[cp].x,       sizeof(float));
        memcpy(&radarPoint.data[cp * radarPoint.point_step + radarPoint.fields[1].offset], &r_data[cp].y,       sizeof(float));
        memcpy(&radarPoint.data[cp * radarPoint.point_step + radarPoint.fields[2].offset], &r_data[cp].z,       sizeof(float));
        memcpy(&radarPoint.data[cp * radarPoint.point_step + radarPoint.fields[3].offset], &r_data[cp].power,   sizeof(float));
        memcpy(&radarPoint.data[cp * radarPoint.point_step + radarPoint.fields[4].offset], &r_data[cp].doppler, sizeof(float));
    }

    // Finally publish all the points from 4SN
    pub_points.publish(radarPoint);

	return 1;
}

// Change Polar Coordinate to Cartesian Coordinate
float DataHandler::polar2x(float range, float azimuth, float elevation)
{
    float phi = azimuth * pi / 180.0f;          // Azimuth angle(radian)
    float theta = elevation * pi / 180.0f;      // Elevation angle(radian)

    return (range * cosf(theta) * sinf(phi));
}

float DataHandler::polar2y(float range, float azimuth, float elevation)
{
    float phi = azimuth * pi / 180.0f;          // Azimuth angle(radian)
    float theta = elevation * pi / 180.0f;      // Elevation angle(radian)

    return (range * cosf(theta) * cosf(phi));
}

float DataHandler::polar2z(float range, float azimuth, float elevation)
{
    float phi = azimuth * pi / 180.0f;          // Azimuth angle(radian)
    float theta = elevation * pi / 180.0f;      // Elevation angle(radian)

    return (range * sinf(theta));
}