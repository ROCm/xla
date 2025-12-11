/* Copyright 2025 The OpenXLA Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef XLA_TSL_DISTRIBUTED_RUNTIME_COORDINATION_NETWORK_CONFIG_SETUP_H_
#define XLA_TSL_DISTRIBUTED_RUNTIME_COORDINATION_NETWORK_CONFIG_SETUP_H_

#include <string>
#include <functional>

#include "absl/status/status.h"

namespace tsl {

// Utility class for network configuration setup in distributed profiling.
// This encapsulates node address discovery and registration logic that was
// previously embedded in coordination_service_agent.cc.
class NetworkConfigSetup {
 public:
  // Callback for inserting key-value pairs into the coordination service
  using InsertKeyValueFn = std::function<absl::Status(const std::string&, const std::string&)>;
  
  // Register this node's address in the KV store.
  // task_id: The task ID of this node
  // insert_fn: Callback to insert key-value pairs into coordination service
  static absl::Status RegisterNodeAddress(
      int task_id,
      InsertKeyValueFn insert_fn);

 private:
  // Get the hostname of this machine
  static std::string GetHostname();
  
  // Get the local IP address for a given interface
  static std::string GetLocalIP(const std::string& ifname);
  
  // Get list of network interfaces to try
  static std::vector<std::string> GetNetworkInterfaces();
};

}  // namespace tsl

#endif  // XLA_TSL_DISTRIBUTED_RUNTIME_COORDINATION_NETWORK_CONFIG_SETUP_H_

