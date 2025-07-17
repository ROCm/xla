#include <cstdint>
#include <stdio.h>

#include "xla/backends/profiler/gpu/rocp.h"
#include "xla/backends/profiler/gpu/rocprofilersdk_wrapper.h"

#include "rocm/include/rocprofiler-sdk/registration.h"


#define eprintf(...) fprintf (stderr, __VA_ARGS__)
#define debug(...) printf (__VA_ARGS__)


namespace xla {
	namespace profiler {

		namespace {

			void tool_tracing_callback(
					rocprofiler_context_id_t      context,
					rocprofiler_buffer_id_t       buffer_id,
					rocprofiler_record_header_t** headers,
					size_t                        num_headers,
					void*                         user_data,
					uint64_t                      drop_count
					){
				eprintf("callback called\n");
			}


			int tool_init(rocprofiler_client_finalize_t fini_fn, void* tool_data) {
				using namespace xla::profiler::wrap;

				debug("rocptracer tool_init called\n");

				int status = 0;
				if (wrap::rocprofiler_is_initialized(&status) != ROCPROFILER_STATUS_SUCCESS)
				{
					eprintf("rocprofiler_is_initialized call failed\n");
					return -1;
				}
				debug("rocprofiler_is_initialized status=%d\n", status);

				// Gather API names
				//name_info_ = GetCallbackTracingNames();

				// Gather agent info
				//int num_gpus = 0;
				//for (const auto& agent : GetGpuDeviceAgents()) {
				//	LOG(INFO) <<"agent id = " << agent.id.handle 
				//		<< ", dev = " << agent.device_id 
				//		<< ", name = " << (agent.name ? agent.name : "null");
				//	agents_[agent.id.handle] = agent;
				//	if (agent.type == ROCPROFILER_AGENT_TYPE_GPU) {
				//		num_gpus++;
				//	}
				//}

				// setup context and event types we need to intercept
				rocprofiler_context_id_t ctx = {.handle = 0};

				if (wrap::rocprofiler_create_context(&ctx) != ROCPROFILER_STATUS_SUCCESS) {
					eprintf("rocprofiler_context_create failed\n");
					return -1;
				}

				// something to do with kernel name gathering?
				//auto code_object_ops = std::vector<rocprofiler_tracing_operation_t>{
				//	ROCPROFILER_CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER};

				//rocprofiler_configure_callback_tracing_service(
				//		utility_context_,
				//		ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT,
				//		code_object_ops.data(),
				//		code_object_ops.size(),
				//		code_object_callback,
				//		nullptr);

				////rocprofiler_start_context(utility_context_);
				//LOG(INFO) << "rocprofiler start utilityContext";

				// a multiple of the page size, and the gap allows the buffer to absorb bursts of GPU events
				constexpr auto buffer_size_bytes = 8 * 4096;
				constexpr auto buffer_watermark_bytes = 1 * 4096;

				rocprofiler_buffer_id_t buffer;

				auto r = wrap::rocprofiler_create_buffer(ctx,
						buffer_size_bytes,
						buffer_watermark_bytes,
						ROCPROFILER_BUFFER_POLICY_LOSSLESS,
						tool_tracing_callback,
						nullptr,
						&buffer);

				if (r != ROCPROFILER_STATUS_SUCCESS) {
					eprintf("rocprofiler tool_init failed: rocprofiler_create_buffer failed\n");
					return -1;
				}

				rocprofiler_buffer_tracing_kind_t domains[] = {
					ROCPROFILER_BUFFER_TRACING_HIP_RUNTIME_API,
					ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH,
					ROCPROFILER_BUFFER_TRACING_MEMORY_COPY,
				};

				for (auto domain : domains ) {
					auto r = wrap::rocprofiler_configure_buffer_tracing_service(ctx, domain, nullptr, 0, buffer);
					if (r != ROCPROFILER_STATUS_SUCCESS) {
						eprintf("rocprofiler tool_init failed: rocprofiler_configure_buffer_tracing_service failed\n");
						return -1;
					}
				}

				// this should not be necessary, we only need the default thread
				//auto client_thread = rocprofiler_callback_thread_t{};
				//rocprofiler_create_callback_thread(&client_thread);
				//rocprofiler_assign_callback_thread(buffer_, client_thread);


				//{
				//	// for annotations
				//	const rocprofiler_tracing_operation_t* hip_ops = nullptr;
				//	size_t hip_ops_count = 0;

				//	wrap::rocprofiler_configure_callback_tracing_service(
				//			ctx,
				//			ROCPROFILER_CALLBACK_TRACING_HIP_RUNTIME_API,
				//			hip_ops,
				//			hip_ops_count,
				//			[](rocprofiler_callback_tracing_record_t record,
				//				rocprofiler_user_data_t*, void*) {
				//			if (record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER) {
				//			const std::string& annotation = tsl::profiler::AnnotationStack::Get();
				//			if (!annotation.empty()) {
				//			RocmTracer::i()
				//			.collector()
				//			->annotation_map()
				//			->Add(record.correlation_id.internal, annotation);
				//			}
				//			}
				//			},
				//			nullptr);
				//}

				// validate context configuration 
				int valid = 0;
				if (wrap::rocprofiler_context_is_valid(ctx, &valid) != ROCPROFILER_STATUS_SUCCESS) {
					eprintf("rocprofiler tool_init failed: rocprofiler_context_is_valid failed\n");
					return -1;
				}
				// seems like from source that a valid context returns 1
				if (valid != 1) {
					eprintf("rocprofiler tool_init failed: rocprofiler context was not valid\n");
					return -1;
				}

				// context is valid so load into singleton
				auto tracer = RocpTracer::Singleton();
				tracer->context_ = new rocp::Context(ctx);
				tracer->buffer_ = new rocp::Buffer(buffer);

				return 0;
			}

			void tool_fini(void* tool_data) {
				debug("rocptracer tool_fini called\n");
			}

			rocprofiler_tool_configure_result_t* rocprofiler_configure(uint32_t version, const char* runtime_version, uint32_t priority, rocprofiler_client_id_t* client) {
				debug("rocprofiler_configure called\n");

				static rocprofiler_tool_configure_result_t r = {
					.size = sizeof(rocprofiler_tool_configure_result_t),
					.initialize = tool_init,
					.finalize = tool_fini,
					.tool_data = nullptr,
				};
				return &r;
			}

			void force_configure() {
				if (xla::profiler::wrap::rocprofiler_force_configure(rocprofiler_configure)) {
					debug("force_configure failed\n");
				}
			}

		} // namespace

		RocpTracer* RocpTracer::Singleton(){
			static auto* i = new RocpTracer();
			debug("rocptracer GetSingleton\n");
			return i;
		}

		bool RocpTracer::IsAvailable() const {
			return is_avail;
		}

		void RocpTracer::Enable(const RocmTracerOptions& options, RocmTraceCollector* collector) {
			debug("rocptracer enable\n");
			is_avail = false;
			//force_configure();

			if (context_) {
				debug("context start\n");
				if (wrap::rocprofiler_start_context(context_->context_id)) {
					eprintf("error during RocpTracer::Enable: rocprofiler_start_context failed\n");
				}
			}
		}

		void RocpTracer::Disable() {
			debug("rocptracer disable\n");
			is_avail = true;
		}

		void RocpTracer::Export(XSpace *space) {
			debug("rocptracer Export\n");
		}

	} // profiler
} // xla

void __attribute__((constructor)) init_rocm_lib () {
	xla::profiler::force_configure();
}


#undef debug
#undef eprintf
