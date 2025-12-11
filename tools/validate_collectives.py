#!/usr/bin/env python3

import argparse
import os
import sys
import json
import csv
import pandas as pd
from typing import List, Dict, Any

# Ensure we can import xplane_pb2 from the same directory
sys.path.append(os.path.dirname(__file__))

try:
    import xplane_pb2
except ImportError:
    print("Error: Could not import xplane_pb2. Please ensure tools/xplane_pb2.py exists.")
    sys.exit(1)

def get_rccl_hlo_df(xspace: xplane_pb2.XSpace) -> pd.DataFrame:
    """
    Extracts RCCL collective events from the XSpace and returns a DataFrame.
    Schema: [node_id, hlo_op, start_ns, end_ns, duration_ns, device_id, correlation_id]
    """
    records = []
    
    # Identify node from hostname if available, otherwise assume 0 or derived from plane names
    # But wait, this tool accepts multiple raw traces OR a combined trace.
    # If combined trace, planes have /node:<id> prefixes or name modifications.
    # If raw trace, we might pass node_id as arg or infer.
    # Let's assume for now we are processing a SINGLE xspace (either raw or combined).
    # If it is combined, we need to parse node_id from plane names.
    # If it is raw, we might assign a default node_id or rely on CLI.
    
    # Actually, the requirement says "Load one or more XSpace files".
    # If multiple files, we assign node_id = index in list? Or file name?
    # Let's handle that in main(). This function processes ONE XSpace.
    # If the XSpace is combined, it will have multiple nodes inside.
    
    for plane in xspace.planes:
        node_id = 0
        # Try to infer node_id from plane name for combined traces
        # Combined Host: /node:<id>/... or /device:GPU:<10000*id>...
        # Combined Device: /device:GPU:<id*64 + ordinal>
        
        # Heuristic for combined traces:
        if "node" in plane.name:
            try:
                # Example: /node:1/device:GPU:0
                parts = plane.name.split('/')
                for p in parts:
                    if p.startswith("node:"):
                        node_id = int(p.split(':')[1])
                        break
            except:
                pass
        elif plane.name.startswith("/device:GPU:"):
            try:
                # Combined GPU plane ID logic: global_id = node_id * 64 + local_id
                # We can't perfectly reverse this without knowing stride, but 64 is standard.
                # However, trace_combiner uses 100 + node*64 + id.
                # Let's try to deduce from ID if name parsing fails.
                gid = plane.id
                if gid >= 10000:
                    # Synthetic host plane: 10000 * node_id
                    # We ignore these for collective validation, as they are host events
                    # and we only care about GPU kernels for happens-before checks.
                    continue
                elif gid >= 100:
                    # GPU plane: 100 + node*64 + local
                    node_id = (gid - 100) // 64
            except:
                pass

        # We only care about Device planes for RCCL kernels usually?
        # Or Host planes for API calls? 
        # Usually happens-before checks are done on the GPU kernels (ncclKernel, etc.)
        # or the Thunk execution on host. 
        # Let's look for GPU planes.
        if not plane.name.startswith("/device:GPU:"):
            continue
            
        # Stat Metadata Map
        stat_map = {m.id: m.name for m in plane.stat_metadata.values()}
        event_map = {m.id: m.name for m in plane.event_metadata.values()}
        
        # Find "hlo_op" stat ID and "correlation_id" stat ID
        hlo_op_id = None
        corr_id_stat_id = None
        for k, v in stat_map.items():
            if v == "hlo_op":
                hlo_op_id = k
            elif v == "correlation_id":
                corr_id_stat_id = k
        
        # If we can't find hlo_op, maybe we check for "nccl" or "rccl" in event name
        
        for line in plane.lines:
            line_base = line.timestamp_ns
            
            for event in line.events:
                name = event_map.get(event.metadata_id, "")
                
                # Filter for Collective Ops
                # 1. Has "hlo_op" stat?
                hlo_op_name = None
                corr_id = -1
                
                for stat in event.stats:
                    if hlo_op_id is not None and stat.metadata_id == hlo_op_id:
                        hlo_op_name = stat.str_value
                    if corr_id_stat_id is not None and stat.metadata_id == corr_id_stat_id:
                        if stat.HasField("uint64_value"):
                            corr_id = stat.uint64_value
                        elif stat.HasField("int64_value"):
                            corr_id = stat.int64_value
                
                # 2. Name contains rccl/nccl?
                # We want strict filtering to avoid mixing non-collectives that have hlo_op.
                is_collective = "rccl" in name.lower() or "nccl" in name.lower() or "all-reduce" in name.lower() or "all-gather" in name.lower()
                
                # Also allow events where the HLO op name strongly suggests a collective
                if not is_collective and hlo_op_name:
                    clean_hlo = hlo_op_name.lower()
                    if any(x in clean_hlo for x in ["reduce", "gather", "scatter", "all-to-all", "send", "recv", "nccl", "rccl"]):
                        is_collective = True
                
                if not is_collective:
                    continue
                    
                # If we have hlo_op, use it as the grouping key. 
                # If not, but name looks like collective, use name.
                key = hlo_op_name if hlo_op_name else name
                
                # We need clean HLO names (e.g. "all-reduce-start.1")
                # Sometimes they are decorated.
                
                offset_ns = (event.offset_ps / 1000.0) if event.HasField('offset_ps') else 0.0
                start_ns = line_base + offset_ns
                duration_ns = event.duration_ps / 1000.0
                end_ns = start_ns + duration_ns
                
                records.append({
                    "node_id": node_id,
                    "plane_id": plane.id,
                    "op_name": key,
                    "start_ns": start_ns,
                    "end_ns": end_ns,
                    "duration_ns": duration_ns,
                    "correlation_id": corr_id
                })

    return pd.DataFrame(records)

def check_collective_happens_before_violations(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyzes the DataFrame for HB violations.
    Returns a summary dictionary.
    """
    if df.empty:
        return {"violations": 0, "overlaps": 0, "warnings": 0, "details": [], "csv_rows": []}

    violations = 0
    overlaps = 0
    warnings = 0
    details = []
    gap_sum_ns = 0.0
    gap_count = 0
    csv_rows = []  # For CSV export

    # Group by Operation Name
    grouped = df.groupby("op_name")
    
    for op_name, group in grouped:
        # We need to compare executions of the SAME collective op across nodes.
        # Assumption: The N-th occurrence of "all-reduce.1" on Node A corresponds 
        # to the N-th occurrence on Node B.
        
        # Sort by start time within each node to establish "N-th occurrence"
        nodes = group["node_id"].unique()
        if len(nodes) < 2:
            continue
        
        # Sort nodes for consistent column ordering
        sorted_nodes = sorted(nodes)
            
        node_queues = {}
        for node in sorted_nodes:
            node_df = group[group["node_id"] == node].sort_values("start_ns")
            node_queues[node] = node_df.to_dict('records')
            
        # Iterate through ranks/occurrences
        # Find max occurrences across nodes
        max_count = max(len(q) for q in node_queues.values())
        
        for i in range(max_count):
            # Gather the i-th instance from each node (if exists)
            instances = []
            for node in sorted_nodes:
                if i < len(node_queues[node]):
                    instances.append(node_queues[node][i])
            
            if len(instances) < 2:
                continue
                
            # Pairwise comparison
            # A collective op is valid if all nodes overlap in time.
            # Violation if: Intersection is empty.
            # i.e., max(start_times) > min(end_times)
            
            starts = [x["start_ns"] for x in instances]
            ends = [x["end_ns"] for x in instances]
            
            latest_start = max(starts)
            earliest_end = min(ends)
            min_start = min(starts)
            latest_end = max(ends)
            
            start_gap = latest_start - min_start
            end_gap = latest_end - earliest_end
            # start_gap = end_gap
            gap_sum_ns += start_gap
            gap_count += 1
            
            # Build CSV row: start_gap, node_0_corr_id, node_1_corr_id, ...
            csv_row = {
                "event_idx": gap_count - 1,
                "op_name": op_name,
                "occurrence": i,
                "start_gap_ns": start_gap
            }
            for inst in instances:
                node = inst["node_id"]
                csv_row[f"node_{node}_corr_id"] = inst["correlation_id"]
                csv_row[f"node_{node}_start_ns"] = inst["start_ns"]
            csv_rows.append(csv_row)
            
            if latest_start < earliest_end:
                overlaps += 1
            else:
                # Violation!
                violations += 1
                
                # Find the culprits (pairs that don't overlap)
                # For reporting, just dump the range
                violation_detail = {
                    "op": op_name,
                    "rank": i,
                    "latest_start_node": [x["node_id"] for x in instances if x["start_ns"] == latest_start][0],
                    "latest_start_time": latest_start,
                    "earliest_end_node": [x["node_id"] for x in instances if x["end_ns"] == earliest_end][0],
                    "earliest_end_time": earliest_end,
                    "gap_ns": latest_start - earliest_end,
                    "correlation_ids": {x["node_id"]: x["correlation_id"] for x in instances}
                }
                details.append(violation_detail)

    avg_gap_ns = gap_sum_ns / gap_count if gap_count > 0 else 0.0

    return {
        "violations": violations,
        "overlaps": overlaps,
        "warnings": warnings,
        "avg_gap_ns": avg_gap_ns,
        "gap_count": gap_count,
        "details": details,
        "csv_rows": csv_rows
    }

def analyze_collective_operations(trace_files: List[str], output_file: str = None, csv_file: str = None):
    all_dfs = []
    
    # If we are given multiple files, we treat them as separate nodes (unless they are combined traces).
    # If 1 file, assume it's combined or single node.
    
    combined_mode = False
    if len(trace_files) == 1:
        # Check if it has multiple nodes inside?
        # get_rccl_hlo_df logic handles extracting node_id from combined trace planes.
        combined_mode = True
        
    for i, fpath in enumerate(trace_files):
        print(f"Loading {fpath}...")
        try:
            with open(fpath, 'rb') as f:
                xspace = xplane_pb2.XSpace()
                xspace.ParseFromString(f.read())
                
            df = get_rccl_hlo_df(xspace)
            
            if not combined_mode:
                # Force node_id to be the file index if we are processing separate raw files
                # and the extraction didn't find explicit node IDs.
                # (Unless extraction logic already found them?)
                # Let's verify.
                unique_nodes = df["node_id"].unique()
                if len(unique_nodes) <= 1 and unique_nodes[0] == 0:
                     df["node_id"] = i
            
            all_dfs.append(df)
            
        except Exception as e:
            print(f"Failed to process {fpath}: {e}")

    if not all_dfs:
        print("No data found.")
        return

    full_df = pd.concat(all_dfs, ignore_index=True)
    print(f"Extracted {len(full_df)} collective events.")
    
    summary = check_collective_happens_before_violations(full_df)
    
    print(f"Analysis Complete:")
    print(f"  Collectives Processed (Groups): {summary.get('gap_count', 0)}")
    print(f"  Violations: {summary['violations']}")
    print(f"  Overlaps (Success): {summary['overlaps']}")
    print(f"  Avg Start Gap: {summary['avg_gap_ns']:.2f} ns")
    
    if output_file:
        # Remove csv_rows from JSON output (it's for CSV only)
        json_summary = {k: v for k, v in summary.items() if k != "csv_rows"}
        with open(output_file, 'w') as f:
            # Convert numpy types to python types for JSON serialization
            def convert(o):
                if isinstance(o, (pd.NA, pd.NaT)):
                    return None
                if hasattr(o, 'item'): 
                    return o.item()
                return o
                
            json.dump(json_summary, f, indent=2, default=convert)
        print(f"Detailed report written to {output_file}")
    
    # Export CSV if requested
    if csv_file and summary.get("csv_rows"):
        export_start_gap_csv(summary["csv_rows"], csv_file)

def export_start_gap_csv(csv_rows: List[Dict], csv_path: str):
    """
    Export collective start gaps to CSV.
    Each row: event_idx, op_name, occurrence, start_gap_ns, node_0_corr_id, node_0_start_ns, node_1_corr_id, node_1_start_ns, ...
    """
    if not csv_rows:
        print("No collective events to export to CSV.")
        return
    
    # Collect all unique column names from all rows
    all_columns = set()
    for row in csv_rows:
        all_columns.update(row.keys())
    
    # Order columns: fixed columns first, then node columns sorted
    fixed_cols = ["event_idx", "op_name", "occurrence", "start_gap_ns"]
    node_cols = sorted([c for c in all_columns if c not in fixed_cols])
    header = fixed_cols + node_cols
    
    try:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=header, extrasaction='ignore')
            writer.writeheader()
            for row in csv_rows:
                # Convert any numpy types to native Python
                clean_row = {}
                for k, v in row.items():
                    if hasattr(v, 'item'):
                        clean_row[k] = v.item()
                    else:
                        clean_row[k] = v
                writer.writerow(clean_row)
        print(f"Exported {len(csv_rows)} collective event gaps to {csv_path}")
    except Exception as e:
        print(f"Error writing CSV: {e}")

def main():
    parser = argparse.ArgumentParser(description="Validate collective operation timing across nodes.")
    parser.add_argument("--traces", nargs='+', required=True, help="List of XSpace files (raw per-node or combined).")
    parser.add_argument("--output", help="JSON output file for violation report.")
    parser.add_argument("--csv", help="CSV output file for start gap per collective event.")
    
    args = parser.parse_args()
    
    analyze_collective_operations(args.traces, args.output, args.csv)

if __name__ == "__main__":
    main()
