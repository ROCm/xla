#!/usr/bin/env python3

import argparse
import bisect
import json
import sys
import os
import math
from typing import Dict, List, Tuple, Any
from tqdm import tqdm

# Ensure we can import xplane_pb2 from the same directory
sys.path.append(os.path.dirname(__file__))

try:
    import xplane_pb2
except ImportError:
    print("Error: Could not import xplane_pb2. Please ensure tools/xplane_pb2.py exists.")
    sys.exit(1)

class LinearInterpolator:
    def __init__(self, points: List[Tuple[float, float]]):
        """
        points: List of (x, y) pairs.
        """
        # Sort by x
        self.points = sorted(points, key=lambda p: p[0])
        self.xs = [p[0] for p in self.points]
        self.ys = [p[1] for p in self.points]

    def map_x_to_y(self, x: float) -> float:
        if not self.points:
            return x

        # Special case: single point - assume slope=1 (no drift)
        # y = y0 + (x - x0) = x + (y0 - x0)
        if len(self.points) == 1:
            x0, y0 = self.points[0]
            return x + (y0 - x0)

        idx = bisect.bisect_right(self.xs, x)
        
        if idx == 0:
            # Extrapolate using first segment
            p0, p1 = self.points[0], self.points[1]
        elif idx >= len(self.points):
            # Extrapolate using last segment
            p0, p1 = self.points[-2], self.points[-1]
        else:
            p0, p1 = self.points[idx-1], self.points[idx]
            
        if p1[0] == p0[0]: return float(p0[1])
        
        slope = (p1[1] - p0[1]) / (p1[0] - p0[0])
        return p0[1] + slope * (x - p0[0])

class SnapshotInterpolator:
    def __init__(self, snapshots: List[Tuple[float, float]]):
        """
        snapshots: list of (sys_ns, tracer_ns)
        """
        # Map R (Device/Tracer) -> S (System/Host)
        # x = tracer_ns, y = sys_ns
        self.r_to_s = LinearInterpolator([(r, s) for s, r in snapshots])
        
        # Map S (System/Host) -> R (Device/Tracer)
        # x = sys_ns, y = tracer_ns
        self.s_to_r = LinearInterpolator([(s, r) for s, r in snapshots])
        
    def map_r_to_s(self, r_time):
        return self.r_to_s.map_x_to_y(r_time)

    def map_s_to_r(self, s_time):
        return self.s_to_r.map_x_to_y(s_time)

class OffsetInterpolator:
    def __init__(self, offset_entries: List[Dict[str, Any]], node_id: int):
        """
        Builds interpolator for Node i -> Node 0.
        Input entries contain (M_0k, Offset).
        M_ik = M_0k + Offset.
        We map M_ik -> M_0k.
        """
        points = []
        for data in offset_entries:
            if data.get("node_id") != node_id:
                continue
            
            # Determine M_0k (Reference Midpoint)
            mid_ref = None
            if "midpoint_sys_ns" in data:
                mid_ref = float(data["midpoint_sys_ns"])
            elif "window_start_ns" in data and "window_end_ns" in data:
                mid_ref = (float(data["window_start_ns"]) + float(data["window_end_ns"])) / 2.0
            
            if mid_ref is None:
                continue
                
            offset = float(data.get("offset_ns", 0.0))
            
            # M_ik = M_0k + Offset
            mid_local = mid_ref + offset
            
            # Pair: (Local, Ref) -> (x, y)
            points.append((mid_local, mid_ref))
            
        self.interp = LinearInterpolator(points)

    def map_local_to_ref(self, local_time):
        if not self.interp.points:
            return local_time
        return self.interp.map_x_to_y(local_time)

def annotate_host_line(line, node_id):
    """Prefix line names to include node id for clarity."""
    prefix = f"/node:{node_id}"
    if line.name:
        line.name = f"{prefix}/{line.name}"
    else:
        line.name = f"{prefix}/line_{line.id}"
    if line.display_name:
        line.display_name = f"Node {node_id}: {line.display_name}"
    else:
        line.display_name = f"Node {node_id}: line_{line.id}"
    # Set display_id to id for proper rendering
    line.display_id = line.id


def annotate_host_plane_lines(plane, node_id):
    for line in plane.lines:
        annotate_host_line(line, node_id)


def rename_gpu_plane(plane, node_id):
    """Rename GPU plane to /device:GPU:<global_id> with unique IDs."""
    try:
        parts = plane.name.split(':')
        if len(parts) >= 3:
            gpu_id = int(parts[2])
        else:
            gpu_id = plane.id
    except ValueError:
        gpu_id = plane.id

    # Use base 100 for GPU planes to avoid collision with /host:CPU (ID=0)
    # Node 0 GPU 0 -> ID=100, Node 0 GPU 1 -> ID=101
    # Node 1 GPU 0 -> ID=164, Node 1 GPU 1 -> ID=165
    new_id = 100 + (node_id * 64) + gpu_id
    plane.name = f"/device:GPU:{new_id}"
    plane.id = new_id
    
    # Remap line IDs and display_ids to avoid clashes across nodes
    # Line IDs like "Stream #1" (ID=1) exist on all nodes and would clash
    # Both id and display_id need to be unique to avoid rendering issues
    line_id_offset = node_id * 1000000  # Large offset to avoid collisions
    for line in plane.lines:
        if node_id > 0:
            line.id = line.id + line_id_offset
        # Always set display_id to match id (ensures uniqueness and proper rendering)
        line.display_id = line.id


def rename_misc_plane(plane, node_id, misc_index):
    """Rename misc planes with unique IDs.
    
    Args:
        plane: The plane to rename
        node_id: The node ID (0, 1, 2, ...)
        misc_index: A unique index for this misc plane within the node (0, 1, 2, ...)
    """
    if plane.name.startswith("/device:GPU:"):
        return
    safe_name = plane.name.lstrip('/')
    # Use base 50000 for misc planes to avoid collision with GPU planes (0-63 per node)
    # and host-as-GPU planes (10000, 20000, ...)
    # Formula: 50000 + node_id * 100 + misc_index
    new_id = 50000 + (node_id * 100) + misc_index
    plane.name = f"/device:GPU:{new_id}_misc_{safe_name}"
    plane.id = new_id
    
    # Remap line IDs and display_ids
    line_id_offset = node_id * 1000000
    for line in plane.lines:
        if node_id > 0:
            line.id = line.id + line_id_offset
        line.display_id = line.id


def rename_host_plane_as_gpu(plane, node_id):
    """Repurpose a host plane (for node > 0) as a synthetic GPU plane."""
    annotate_host_plane_lines(plane, node_id)
    new_id = node_id * 10000
    plane.name = f"/device:GPU:{new_id}"
    plane.id = new_id
    
    # Remap line IDs to avoid clashes (node > 0 always for this function)
    line_id_offset = node_id * 1000000
    for line in plane.lines:
        line.id = line.id + line_id_offset
        line.display_id = line.id


def merge_host_plane(source_plane, target_plane, node_id):
    """Merge host plane from another node into the canonical host plane."""
    src = xplane_pb2.XPlane()
    src.ParseFromString(source_plane.SerializeToString())
    annotate_host_plane_lines(src, node_id)

    # Compute offsets for metadata and lines to avoid collisions.
    event_id_offset = (max(target_plane.event_metadata.keys())
                       if target_plane.event_metadata else 0) + 1
    stat_id_offset = (max(target_plane.stat_metadata.keys())
                      if target_plane.stat_metadata else 0) + 1
    line_id_counter = (max((line.id for line in target_plane.lines), default=-1) + 1)

    event_id_map = {}
    for key, meta in src.event_metadata.items():
        new_key = key + event_id_offset
        event_id_map[key] = new_key
        target_meta = target_plane.event_metadata[new_key]
        target_meta.CopyFrom(meta)
        target_meta.id = new_key

    stat_id_map = {}
    for key, meta in src.stat_metadata.items():
        new_key = key + stat_id_offset
        stat_id_map[key] = new_key
        target_meta = target_plane.stat_metadata[new_key]
        target_meta.CopyFrom(meta)
        target_meta.id = new_key

    # Plane-level stats
    for stat in src.stats:
        new_stat = target_plane.stats.add()
        new_stat.CopyFrom(stat)
        if new_stat.metadata_id in stat_id_map:
            new_stat.metadata_id = stat_id_map[new_stat.metadata_id]

    # Copy lines and fix references.
    for line in src.lines:
        new_line = target_plane.lines.add()
        new_line.CopyFrom(line)
        new_line.id = line_id_counter
        new_line.display_id = line_id_counter  # Also set display_id to avoid rendering clashes
        line_id_counter += 1

        for event in new_line.events:
            if event.metadata_id in event_id_map:
                event.metadata_id = event_id_map[event.metadata_id]
            for stat in event.stats:
                if stat.metadata_id in stat_id_map:
                    stat.metadata_id = stat_id_map[stat.metadata_id]

def load_snapshot_pairs(filepath):
    try:
        pairs = xplane_pb2.SnapshotPairs()
        with open(filepath, 'rb') as f:
            pairs.ParseFromString(f.read())
        return [(p.sys_clock_ns, p.tracer_clock_ns) for p in pairs.pairs]
    except Exception as e:
        print(f"Error loading snapshots from {filepath}: {e}")
        return []

def extract_snapshots_from_xspace(xspace, start_walltime_ns=0):
    """Extract snapshots from the embedded /host:snapshots plane.
    
    Args:
        xspace: The XSpace protobuf
        start_walltime_ns: The absolute start walltime. If provided and the extracted
                          sys_ns values appear to be relative (small), they will be
                          converted to absolute by adding this value.
    """
    snapshots = []
    for plane in xspace.planes:
        if plane.name == "/host:snapshots":
            # Create a map of metadata ID to stat name for this plane
            stat_metadata_map = {}
            for sm in plane.stat_metadata.values():
                stat_metadata_map[sm.id] = sm.name
                
            for line in plane.lines:
                line_base_ns = line.timestamp_ns
                for event in line.events:
                    # Calculate sys_ns (S time)
                    offset_ns = 0
                    if event.HasField('offset_ps'):
                        offset_ns = event.offset_ps / 1000.0
                    sys_ns = float(line_base_ns) + offset_ns
                    
                    # Find rocm_ns (R time) in stats
                    rocm_ns = None
                    for stat in event.stats:
                        if stat.metadata_id in stat_metadata_map:
                            name = stat_metadata_map[stat.metadata_id]
                            if name == "rocm_ns":
                                if stat.HasField('uint64_value'):
                                    rocm_ns = stat.uint64_value
                                elif stat.HasField('int64_value'):
                                    rocm_ns = stat.int64_value
                                break
                    
                    if rocm_ns is not None:
                        snapshots.append((sys_ns, float(rocm_ns)))
            break
    
    # Check if sys_ns values are relative (much smaller than rocm_ns values)
    # If so, convert to absolute by adding start_walltime_ns
    if snapshots and start_walltime_ns > 0:
        # Check first snapshot: if sys_ns << rocm_ns, it's likely relative
        first_sys, first_rocm = snapshots[0]
        if first_sys < 1e15 and first_rocm > 1e15:
            # sys_ns appears to be relative, convert to absolute
            snapshots = [(sys_ns + start_walltime_ns, rocm_ns) for sys_ns, rocm_ns in snapshots]
    
    # Sort by tracer time (R)
    if snapshots:
        snapshots.sort(key=lambda x: x[1])
        
    return snapshots

def parse_offset_file(filepath: str) -> Tuple[Dict[int, Dict[str, Any]], List[Dict[str, Any]]]:
    """Parse the JSONL offsets file once. Returns (meta_nodes, entries)."""
    meta_nodes: Dict[int, Dict[str, Any]] = {}
    entries: List[Dict[str, Any]] = []
    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if data.get("meta"):
                    for node in data.get("nodes", []):
                        node_id = int(node["node_id"])
                        meta_nodes[node_id] = node
                else:
                    entries.append(data)
    except Exception as e:
        print(f"Error parsing offsets file {filepath}: {e}")
    return meta_nodes, entries

def process_trace_file(trace_path, snapshot_path, offset_entries, meta_nodes, node_id,
                       apply_correction=True, print_events=False, print_collectives=False,
                        event_node_dict=None):
    print(f"Processing node {node_id}: trace={trace_path}")
    
    if event_node_dict is not None and node_id not in event_node_dict:
        event_node_dict[node_id] = []
    
    # 1. Load XSpace
    xspace = xplane_pb2.XSpace()
    try:
        with open(trace_path, 'rb') as f:
            content = f.read()
            print(f"  Read {len(content)} bytes from {trace_path}")
            xspace.ParseFromString(content)
            print(f"  Parsed XSpace: {len(xspace.planes)} planes")
    except Exception as e:
        print(f"Failed to read trace {trace_path}: {e}")
        return None

    # 2. Load Snapshots
    snap_interp = None
    offset_interp = None
    
    if not apply_correction:
        print(f"  Info: Correction disabled for node {node_id}; leaving timestamps unchanged.")
    else:
        snapshots = []
        if snapshot_path:
            snapshots = load_snapshot_pairs(snapshot_path)
            print(f"  Loaded {len(snapshots)} snapshots from file {snapshot_path}")
        
        if not snapshots:
            print(f"  Info: No separate snapshot file or empty. Attempting to extract from XSpace plane '/host:snapshots'...")
            # Pass start_walltime_ns to convert relative sys_ns to absolute if needed
            node_meta = meta_nodes.get(node_id, {})
            start_wall_for_snap = float(node_meta.get("start_walltime_ns", 0))
            snapshots = extract_snapshots_from_xspace(xspace, start_wall_for_snap)
            print(f"  Extracted {len(snapshots)} snapshots from XSpace")
            
        if len(snapshots) < 2:
            print(f"  Warning: Not enough snapshots for node {node_id}. Skipping timestamp correction.")
        else:
            print(f"  Info: Found {len(snapshots)} snapshot pairs for node {node_id}. Interpolation enabled.")
            # Debug: print first and last snapshot
            print(f"    Snapshot[0]: sys_ns={snapshots[0][0]:.0f}, rocm_ns={snapshots[0][1]:.0f}")
            print(f"    Snapshot[-1]: sys_ns={snapshots[-1][0]:.0f}, rocm_ns={snapshots[-1][1]:.0f}")
            snap_interp = SnapshotInterpolator(snapshots)

        # 3. Load Offsets
        offset_interp = OffsetInterpolator(offset_entries, node_id)
        if not offset_interp.interp.points:
             print(f"  Warning: No offsets for node {node_id}. Using 0 offset.")
        else:
             print(f"  Info: Found {len(offset_interp.interp.points)} offset windows for node {node_id}.")

    # Start times from meta
    node_meta = meta_nodes.get(node_id, {})
    # S_hi (Start Walltime Node i)
    start_wall = float(node_meta.get("start_walltime_ns", 0))
    # S_di (Start GPU Time Node i)
    start_gpu = float(node_meta.get("start_gpu_ns", 0))
    
    # S_h0 (Start Walltime Node 0)
    ref_meta = meta_nodes.get(0, {})
    start_wall_0 = float(ref_meta.get("start_walltime_ns", 0))
    if start_wall_0 == 0:
        start_wall_0 = start_wall
    
    print(f"  Meta for node {node_id}: start_wall={start_wall:.0f}, start_gpu={start_gpu:.0f}")
    print(f"  Reference (node 0): start_wall_0={start_wall_0:.0f}")

    # Process Planes
    events_processed = 0
    collective_events_printed = 0
    for plane in xspace.planes:
        # Determine if this is a Device plane or Host plane
        is_device_plane = plane.name.startswith("/device:GPU:")
        
        # plane_base_in: The base timestamp used to unnormalize events
        # Host: S_hi
        # Device: S_di
        plane_base_in = start_gpu if is_device_plane else start_wall
        
        # Build stat metadata map for this plane to find correlation_id and hlo_op
        stat_map = {m.id: m.name for m in plane.stat_metadata.values()}
        event_map = {m.id: m.name for m in plane.event_metadata.values()}
        
        corr_id_stat_id = None
        hlo_op_stat_id = None
        for k, v in stat_map.items():
            if v == "correlation_id":
                corr_id_stat_id = k
            elif v == "hlo_op":
                hlo_op_stat_id = k
        
        # Iterate Lines
        for line in plane.lines:
            last_timestamp = -1
            corrected_data = []
            
            # Original timestamps in file are relative to line.timestamp_ns, 
            # but logically correspond to plane_base_in + offset.
            
            for event_idx, event in tqdm(enumerate(line.events), total=len(line.events), desc=f"Processing events for node {node_id}"):
                events_processed += 1
                
                # Extract correlation_id and hlo_op for collective detection
                corr_id = -1
                hlo_op_name = None
                for stat in event.stats:
                    if corr_id_stat_id is not None and stat.metadata_id == corr_id_stat_id:
                        if stat.HasField("uint64_value"):
                            corr_id = stat.uint64_value
                        elif stat.HasField("int64_value"):
                            corr_id = stat.int64_value
                    if hlo_op_stat_id is not None and stat.metadata_id == hlo_op_stat_id:
                        hlo_op_name = stat.str_value
                
                # Check if this is a collective event
                # breakpoint()
                event_name = event_map.get(event.metadata_id, "")
                is_collective = False
                if "rccl" in event_name.lower() or "nccl" in event_name.lower():
                    is_collective = True
                elif hlo_op_name:
                    clean_hlo = hlo_op_name.lower()
                    if any(x in clean_hlo for x in ["reduce", "gather", "scatter", "all-to-all"]):
                        is_collective = True
                
                # 1. Unnormalize
                # E_un = Base + Offset
                offset_ns = (event.offset_ps / 1000.0) if event.HasField('offset_ps') else 0.0
                e_un = plane_base_in + offset_ns
                
                e_final = e_un
                
                # Apply Algorithm
                interpolated_offset = 0.0
                e_h0 = e_un  # Default: no correction means e_h0 == e_un
                if snap_interp and offset_interp and offset_interp.interp.points:
                    # Full correction with snapshot interpolation and offset adjustment
                    if event.metadata_id == 2:
                        # breakpoint()
                        pass
                    if is_device_plane:
                        # Device Event Transform
                        # E_di = E_un
                        if is_collective and node_id > 0:
                            # breakpoint()
                            pass
                        e_di = e_un
                        
                        # map R -> S (Device i -> Host i)
                        e_hi = snap_interp.map_r_to_s(e_di)
                        # e_hi = e_un
                        
                        # map Host i -> Host 0 (using M_ik -> M_0k)
                        e_h0 = offset_interp.map_local_to_ref(e_hi)
                        interpolated_offset = e_hi - e_h0  # The offset applied
                        
                        # map Host 0 -> Device i (Inverse Snapshot)
                        e_di_c = snap_interp.map_s_to_r(e_h0)
                        
                        # Target: Corrected Device Time
                        # e_final = e_di_c # HACK
                        e_final = e_h0
                        
                        # Debug: print first device event transformation
                        if events_processed == 1 and is_device_plane:
                            print(f"  DEBUG first device event on node {node_id}:")
                            print(f"    offset_ns={offset_ns:.0f}, e_un={e_un:.0f}")
                            print(f"    e_di={e_di:.0f} -> e_hi={e_hi:.0f} (R->S)")
                            print(f"    e_hi={e_hi:.0f} -> e_h0={e_h0:.0f} (offset)")
                            print(f"    e_final={e_final:.0f} (Host_0 time)")
                        
                    else:
                        # Host Event Transform
                        # E_hi = E_un
                        e_hi = e_un
                        
                        # map Host i -> Host 0
                        e_h0 = offset_interp.map_local_to_ref(e_hi)
                        interpolated_offset = e_hi - e_h0  # The offset applied
                        
                        # Target: Corrected Host Time
                        e_final = e_h0
                else:
                    # No full correction available - do simple walltime alignment
                    # Shift events by the difference in start walltimes so that
                    # events from different nodes are aligned to Node 0's reference time.
                    # 
                    # For an event at offset_ns within Node i's trace:
                    #   Real wall time = start_wall_i + offset_ns
                    #   Time since Node 0's start = start_wall_i + offset_ns - start_wall_0
                    #                             = offset_ns + (start_wall_i - start_wall_0)
                    walltime_shift = start_wall - start_wall_0
                    e_final = start_wall_0 + offset_ns + walltime_shift
                    # Simplifies to: e_final = start_wall + offset_ns (absolute wall time)
                    
                    if events_processed == 1:
                        print(f"  DEBUG no-correction mode on node {node_id}:")
                        print(f"    offset_ns={offset_ns:.0f}, walltime_shift={walltime_shift:.0f}")
                        print(f"    e_final={e_final:.0f} (aligned to Node 0 reference)")
                
                # Step 10: Monotonicity Guard
                if e_final <= last_timestamp:
                    e_final = last_timestamp + 1.0
                
                last_timestamp = e_final
                corrected_data.append((e_final, event))
                
                # Print collective event details if requested
                if print_collectives and is_collective:
                    collective_events_printed += 1
                    op_key = hlo_op_name if hlo_op_name else event_name
                    # e_un is the unnormalized time on this node
                    # e_h0 is the corresponding time on master node (node 0)
                    # For node 0, e_h0 == e_un (no offset applied)
                    event_node_dict[node_id].append((e_un, interpolated_offset, e_final))
                    master_node_time = event_node_dict[0][collective_events_printed - 1][2]
                    print(f"  COLLECTIVE node={node_id} corr_id={corr_id} op='{op_key}' "
                          f"e_un={e_un:.0f}ns e_h0={master_node_time:.0f}ns "
                          f"offset_applied={interpolated_offset:.0f}ns e_final={e_final:.0f}ns")

                if print_events:
                    meta = plane.event_metadata.get(event.metadata_id)
                    meta_name = meta.name if meta else "<unknown>"
                    print(f"    Node {node_id} Plane '{plane.name}' Line {line.id} "
                          f"Event {event_idx}: meta='{meta_name}', ts={e_final:.0f} ns")
            
            # Rewrite line timestamps and offsets
            if corrected_data:
                # All events are now in Host_0 time domain (either via full correction
                # or via simple walltime alignment), so use start_wall_0 as the base.
                plane_base_out = start_wall_0

                # Set line timestamp to 0 to allow relative offsets
                line.timestamp_ns = 0
                
                for (ts, event) in corrected_data:
                    # Calculate offset relative to the appropriate base
                    new_offset_ns = ts - plane_base_out
                    new_offset_ps = int(new_offset_ns * 1000.0)
                    
                    # Clamp non-negative just in case
                    if new_offset_ps < 0: 
                        pass
                    try:
                        event.offset_ps = new_offset_ps
                    except Exception as e:
                        print(f"Error setting offset_ps for event {event.metadata_id}: {e}")
                        # breakpoint()
                        pass
    
    print(f"  Processed {events_processed} events in {len(xspace.planes)} planes.")
    if print_collectives:
        print(f"  Printed {collective_events_printed} collective events.")
    return xspace

def main():
    parser = argparse.ArgumentParser(description="Combine and align multiple XPlane traces.")
    parser.add_argument("--traces", nargs='+', required=True, help="List of xplane.pb trace files")
    parser.add_argument("--snapshots", nargs='+', help="List of snapshot pair files (optional, defaults to extracting from trace)")
    parser.add_argument("--offsets", required=True, help="Path to offsets JSONL file")
    parser.add_argument("--output", required=True, help="Output combined xplane.pb path")
    parser.add_argument("--no_correction", action="store_true",
                        help="Skip timestamp correction and only merge planes/metadata.")
    parser.add_argument("--print_events", action="store_true",
                        help="Print every event's metadata name and timestamp while processing (debug).")
    parser.add_argument("--print_collectives", action="store_true",
                        help="Print collective events with correlation_id and interpolated offset.")
    
    args = parser.parse_args()
    
    snapshot_files = args.snapshots if args.snapshots else [None] * len(args.traces)

    if len(args.traces) != len(snapshot_files):
        print("Error: Number of trace files must match number of snapshot files (if provided).")
        sys.exit(1)
        
    meta_nodes, offset_entries = parse_offset_file(args.offsets)

    combined_xspace = xplane_pb2.XSpace()
    combined_host_plane = None
    total_planes = 0
    misc_plane_counter = 0  # Global counter for unique misc plane IDs
    all_hostnames = []
    
    event_node_dict = {}
    for i, (trace_f, snap_f) in enumerate(zip(args.traces, snapshot_files)):
        
        node_xspace = process_trace_file(
            trace_f, snap_f, offset_entries, meta_nodes, i,
            apply_correction=not args.no_correction,
            print_events=args.print_events,
            print_collectives=args.print_collectives, 
            event_node_dict=event_node_dict)
        if node_xspace:
            # Collect hostnames from each XSpace
            for hostname in node_xspace.hostnames:
                if hostname not in all_hostnames:
                    all_hostnames.append(hostname)
            
            # Merge planes into combined XSpace
            for plane in node_xspace.planes:
                if plane.name == "/host:CPU":
                    if i == 0:
                        if combined_host_plane is None:
                            combined_host_plane = combined_xspace.planes.add()
                            combined_host_plane.ParseFromString(plane.SerializeToString())
                            annotate_host_plane_lines(combined_host_plane, i)
                            total_planes += 1
                            print(f"Using node {i} host plane as canonical /host:CPU")
                        else:
                            merge_host_plane(plane, combined_host_plane, i)
                            print(f"Merged node {i} host plane into canonical host plane")
                    else:
                        new_plane = combined_xspace.planes.add()
                        new_plane.ParseFromString(plane.SerializeToString())
                        rename_host_plane_as_gpu(new_plane, i)
                        total_planes += 1
                        print(f"Mapped node {i} host plane to synthetic device {new_plane.name}")
                elif plane.name.startswith("/device:GPU:"):
                    new_plane = combined_xspace.planes.add()
                    new_plane.ParseFromString(plane.SerializeToString())
                    old_name = new_plane.name
                    rename_gpu_plane(new_plane, i)
                    total_planes += 1
                    print(f"  GPU plane: {old_name} -> {new_plane.name} (ID={new_plane.id}, Lines={len(new_plane.lines)})")
                else:
                    new_plane = combined_xspace.planes.add()
                    new_plane.ParseFromString(plane.SerializeToString())
                    rename_misc_plane(new_plane, i, misc_plane_counter)
                    misc_plane_counter += 1
                    total_planes += 1

    # Add collected hostnames to combined XSpace
    for hostname in all_hostnames:
        combined_xspace.hostnames.append(hostname)
    print(f"Hostnames in combined trace: {all_hostnames}")
    
    print(f"Merged {total_planes} planes into combined XSpace.")
    print("Note: When viewing large combined traces, set 'TF_PROFILER_TRACE_VIEWER_MAX_EVENTS' env var")
    print("      to a large value (e.g. 10000000) to avoid event truncation in the viewer.")

    # Write Output
    try:
        out_data = combined_xspace.SerializeToString()
        print(f"Writing {len(out_data)} bytes to {args.output}")
        with open(args.output, 'wb') as f:
            f.write(out_data)
        print(f"Successfully wrote combined trace to {args.output}")
    except Exception as e:
        print(f"Error writing output: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

