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

#include "xla/tsl/distributed_runtime/coordination/network_config_setup.h"

#include <arpa/inet.h>
#include <ifaddrs.h>
#include <netdb.h>
#include <unistd.h>

#include <cstdlib>
#include <cstring>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "tsl/platform/logging.h"

namespace tsl {

std::string NetworkConfigSetup::GetHostname() {
  char hostname[256];
  hostname[255] = '\0';
  if (gethostname(hostname, 255) == 0) {
    return std::string(hostname);
  }
  return "unknown-host";
}

std::string NetworkConfigSetup::GetLocalIP(const std::string& ifname) {
  struct ifaddrs* ifaddr;
  if (getifaddrs(&ifaddr) == -1) {
    perror("getifaddrs");
    return std::string();
  }
  
  std::string ip;
  for (struct ifaddrs* ifa = ifaddr; ifa != nullptr; ifa = ifa->ifa_next) {
    if (ifa->ifa_addr == nullptr) continue;
    if (ifa->ifa_addr->sa_family == AF_INET && 
        strcmp(ifa->ifa_name, ifname.c_str()) == 0) {
      char buf[INET_ADDRSTRLEN];
      struct sockaddr_in* sa = reinterpret_cast<struct sockaddr_in*>(ifa->ifa_addr);
      if (inet_ntop(AF_INET, &(sa->sin_addr), buf, INET_ADDRSTRLEN) != nullptr) {
        ip = buf;
      }
      break;
    }
  }
  freeifaddrs(ifaddr);
  return ip;
}

std::vector<std::string> NetworkConfigSetup::GetNetworkInterfaces() {
  std::vector<std::string> ifnames;
  const char* env_nics = std::getenv("NCCL_SOCKET_IFNAME");
  if (env_nics != nullptr) {
    ifnames = absl::StrSplit(std::string(env_nics), ',');
  } else {
    ifnames = {"eth0"};
  }
  return ifnames;
}

absl::Status NetworkConfigSetup::RegisterNodeAddress(
    int task_id,
    InsertKeyValueFn insert_fn) {
  std::vector<std::string> ifnames = GetNetworkInterfaces();
  
  for (const auto& ifname : ifnames) {
    std::string ip = GetLocalIP(ifname);
    if (!ip.empty()) {
      std::string addr_key = absl::StrCat("rocm:node_addresses:", task_id);
      std::string address = absl::StrCat(ip, ":8765");
      
      absl::Status s = insert_fn(addr_key, address);
      if (!s.ok()) {
        LOG(ERROR) << "Failed to insert key-value for node address " 
                   << ifname << ": " << s;
      } else {
        LOG(INFO) << "Inserted key-value for node address " 
                  << ifname << ": " << address;
        return absl::OkStatus();
      }
    }
  }
  
  return absl::UnavailableError(
      "Failed to register node address: no suitable network interface found");
}

}  // namespace tsl

