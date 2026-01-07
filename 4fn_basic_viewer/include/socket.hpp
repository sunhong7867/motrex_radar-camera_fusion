#ifndef SOCKET_H_
#define SOCKET_H_

#include <unistd.h>
#include <cstring>
#include "radar_packet.hpp"

class Socket
{
	public :
		Socket() : socket_fd(-1) { }
		Socket(int _socket_fd) : socket_fd(_socket_fd) { }
		~Socket();
		bool connectSocket(const char *ip, int port);
		int readData(uint8_t *buf, uint32_t size, bool isBlocked);
		int getSocketfd() const { return socket_fd;}

	private :
		int socket_fd;
};

#endif  // SOCKET_H_

