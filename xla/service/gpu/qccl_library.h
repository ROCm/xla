#ifndef QCCL_LIB_H
#define QCCL_LIB_H 1

// QCCL = Quick CCL
#include <cstdint>
#include "xla/stream_executor/gpu/gpu_types.h"

namespace xla::gpu {

enum QCCL_Result : uint32_t {
  OK,
  NotInitialized,
  InvalidParams,
  Failed,
};

#define QCCL_OK(res) (res) == QCCL_Result::OK

// possibly lazy init for each node: to be called on a thread with assigned GPU
QCCL_Result qcclInit(uint32_t ID);

// function run on a thread: this ID receives from recvPeer and sends to sendPeer
QCCL_Result qcclSendRecv(uint32_t ID, uint32_t numSubscribedPeers, 
        uint32_t recvPeer, void *recvBuf, size_t recvSize, 
        uint32_t sendPeer, void *sendBuf, size_t sendSize);

// register node ID as being a gateway for sending data from peerStart to peerEnd
QCCL_Result qcclGatewaySend(uint32_t ID, uint32_t numSubscribedPeers, 
        uint32_t peerStart, uint32_t peerEnd, 
        size_t dataOfs, size_t dataSize);

// run previously enqueued send-recv primitives on a stream
QCCL_Result qcclRun(uint32_t ID, ::stream_executor::gpu::GpuStreamHandle stream);

QCCL_Result qcclSyncInit();
QCCL_Result qcclSyncGPUs(::stream_executor::gpu::GpuStreamHandle stream);

}

#endif // QCCL_LIB_H