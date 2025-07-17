
#include "xla/backends/profiler/gpu/rocm_collector.h"
#include "xla/backends/profiler/gpu/rocm_tracer.h"
#include "xla/backends/profiler/gpu/rocprofilersdk_wrapper.h"
#include "tsl/profiler/lib/profiler_interface.h"


namespace xla {
namespace profiler {

namespace rocp {
class Context {
public:
	Context(rocprofiler_context_id_t context) : context_id(context) {}
	rocprofiler_context_id_t context_id;

};

class Buffer {
public:
	Buffer(rocprofiler_buffer_id_t buffer) : buffer_id(buffer) {}
	rocprofiler_buffer_id_t buffer_id;
};
}

class RocpTraceCollector {
public:
	void Export(XSpace *space);
};



class RocpTracer {
public:
	RocpTracer() {}

	static RocpTracer* Singleton();
	bool IsAvailable() const;
	void Enable(const RocmTracerOptions& options, RocmTraceCollector* collector);
	void Disable();
	void Export(XSpace *space);

	// Disable copy and move because singleton pattern.
	RocpTracer(const RocmTracer&) = delete;
	RocpTracer& operator=(const RocmTracer&) = delete;

	rocp::Context* context_ = nullptr;
	rocp::Buffer* buffer_ = nullptr;

private:
	RocmTraceCollector* collector_ = nullptr;
	bool is_avail = true;
};


}}
