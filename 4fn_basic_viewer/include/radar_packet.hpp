#ifndef RADAR_PACKET_H_
#define RADAR_PACKET_H_
#include <cstdint>

#define NETWORK_TX_HEADER_LENGTH           				(4)
#define MAX_BUF_SIZE                					(1920*1080*4)
#define RADAR_CMD_HEADER_LENGTH                         (36U)
#define RADAR_OUTPUT_MAGIC_WORD_LENGTH                  (8U)
#define NETWORK_PACKET_HEADER_LENGTH					(16)
#define MAX_NUM_POINTS_PER_FRAME						(6144)
#define FRAME_SIZE               						(128 * 1024)   // packet size : 128KB
#define RADAR_DATA_RX_PORT     							(29172)

static uint8_t NETWORK_TX_HEADER[NETWORK_TX_HEADER_LENGTH] = {0x21, 0x43, 0xcd, 0xab};
static uint8_t radarMagicWord[RADAR_OUTPUT_MAGIC_WORD_LENGTH] = {1, 2, 3, 4, 5, 6, 7, 8};

inline uint32_t GetU32(const uint8_t *a) { return a[0] | ((uint32_t)a[1] << 8) | ((uint32_t)a[2] << 16) | ((uint32_t)a[3] << 24); }

typedef struct 
{
    /**< Number of buffers */
    unsigned int numBuf;

    /**< Header magic number NETWORK_RX_HEADER */
    unsigned int header;

    /**< Payload type NETWORK_RX_TYPE_* */
    unsigned int payloadType;

    /**< channel ID */
    unsigned int chNum;

    /**< Size of payload data in bytes */
    unsigned int dataSize = 0;

    /**< Width of video frame */
    unsigned int width;

    /**< Height of video frame */
    unsigned int height;

    /**< Pitch of video frame in bytes */
    unsigned int pitch[2];
} NetworkRx_CmdHeader;

typedef struct
{
    uint8_t  magicWord[RADAR_OUTPUT_MAGIC_WORD_LENGTH];
    uint32_t protocol_version;
    uint32_t header_size;
    uint32_t payload_size;
    uint32_t antenna_model;
    uint32_t SBL_version;
    uint32_t firmware_version;
    char     serial_number[16];
    uint64_t main_reserved[16];
    uint32_t frame_counter;
    uint32_t time_stamp[2];
    uint32_t target_info_format_type;
    uint32_t targetNumber;
    uint32_t payload_header_size;
    uint32_t capture_time_stamp;
    uint32_t cfar_detect_target_number;
    uint32_t frame_latency;
    uint32_t payload_reserved;
} packet_header_t;

typedef struct
{
    /* radar index */
    uint8_t position; 

    /* packet size */
    uint32_t size;

    /* TI network header */
    NetworkRx_CmdHeader cmdHeader;

    /* 1 frame data */
    union 
    {
        struct  // Total 131072 bytes
        {
            /* packet header */
            packet_header_t pkHeader;

            /* packet data */
            uint8_t data[MAX_NUM_POINTS_PER_FRAME];

            uint8_t trash[7976];
        };

        /* packet buffer */
        uint8_t buf[FRAME_SIZE];
    };

    /* buffer used */
    bool used;
} packet_buffer_t;

typedef struct 
{
    float       x;
    float       y;
    float       z;
    float       doppler;
    uint32_t    power;
} point_data_t;

typedef struct 
{
    float       data[3];
    float       doppler;
    float       power;
} rcv_data_t;
#endif
