#include <iostream>
#include <sys/ioctl.h>
#include "sys/socket.h"
#include "sys/types.h"
#include "netinet/in.h"
#include "arpa/inet.h"
#include "fcntl.h"
#include "socket.hpp"

Socket::~Socket()
{
    if(socket_fd >= 0) {
        shutdown(socket_fd, SHUT_RDWR);
        close(socket_fd); 
        socket_fd = -1;
    }
}

bool Socket::connectSocket(const char* ip, int port)
{
    struct sockaddr_in addr;
    int ret;
 
    if(socket_fd < 0) {
        socket_fd = socket(PF_INET, SOCK_STREAM, IPPROTO_TCP);
        std::cout << "Socket created(" << socket_fd << ")" << std::endl;
    }

    if(ip == nullptr) {
        std::cout << "Invalide IP address" << std::endl;
        return false;
    }
    
    if(port < 0) {
        std::cout << "Invalid port number" << std::endl;
        return false;
    }

    //set non blocking
    ret = fcntl(socket_fd, F_GETFL, 0);
    fcntl(socket_fd, F_SETFL, ret | O_NONBLOCK);

    memset(&addr, 0, sizeof(addr));

    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = inet_addr(ip);
    addr.sin_port = htons(port);
    
    if(connect(socket_fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        std::cout << "Connection failed !" << std::endl;
        return false;
    }

    return true;
}

int Socket::readData(uint8_t *buf, uint32_t size, bool isBlocked)
{
    int readBytes = 0;
    int packetSize = 0;
    int getSize = 0;

    if(buf == nullptr) {
        std::cout << "buf is nullptr" << std::endl;
        return readBytes;
    }

    if(size == 0) {
        std::cout << "size is not correct" << std::endl;
        return readBytes;
    }
    
    int remainSize = size;
    int readSize = 0;

    do {
        ioctl(socket_fd, FIONREAD, &packetSize);
        if(packetSize > 0) {            
            getSize = read(socket_fd, &buf[readSize], remainSize);
            readSize += getSize;
            remainSize -= getSize;
        }

        if(remainSize == 0) {      
            readBytes = readSize;      
            break;
        }

        /* Non sleep */
        if(isBlocked != true) {
            break;
        }

        usleep(1000); // 1ms
    } while (isBlocked);

    return readBytes;
}
