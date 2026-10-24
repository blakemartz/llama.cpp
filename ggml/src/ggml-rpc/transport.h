#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>
#include <functional>

struct socket_t;
typedef std::shared_ptr<socket_t> socket_ptr;

static constexpr size_t MAX_CHUNK_SIZE = 1024ull * 1024ull * 1024ull; // 1 GiB
static constexpr size_t RPC_CONN_CAPS_SIZE = 24;

struct socket_t {
    ~socket_t();

    bool send_data(const void * data, size_t size);
    bool recv_data(void * data, size_t size);
    // Must be called at every message boundary: the RDMA transport coalesces
    // writes into fixed-size frames and posts the trailing partial frame only
    // here. No-op on TCP.
    bool flush();

    socket_ptr accept();

    void get_caps(uint8_t * local_caps);
    void update_caps(const uint8_t * remote_caps);
    const char * transport_name() const;
    // fork: a background thread keeps reading the (TCP) socket into memory, up to max_bytes ahead, and recv_data
    // serves from it - so a peer can stream the next requests while this end is busy (GGML_RPC_READ_AHEAD_MB)
    void start_read_ahead(size_t max_bytes);
    // fork: a background thread writes queued whole messages (send_owned) so this end can go on executing commands
    // while a big reply streams out; send_data waits for the queue to drain, so replies stay in order (TCP only,
    // GGML_RPC_WRITE_BEHIND)
    void start_write_behind();
    bool write_behind_active() const;
    void stop_write_behind();                       // drain + join (before the hooks' owners go away)
    std::vector<uint8_t> wb_acquire();              // a recycled message buffer (capacity kept), or empty
    bool send_owned(std::vector<uint8_t> && msg);   // msg = the complete wire bytes of one message
    // head, then size bytes at data, once wait() returns (on the writer thread); done() runs after the send
    bool send_deferred(std::vector<uint8_t> && head, const void * data, size_t size,
                       std::function<void()> wait, std::function<void()> done);

    static socket_ptr create_server(const char * host, int port);
    static socket_ptr connect(const char * host, int port);

private:
    struct impl;
    explicit socket_t(std::unique_ptr<impl> p);
    std::unique_ptr<impl> pimpl;
};

bool rpc_transport_init();
void rpc_transport_shutdown();
