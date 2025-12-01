#!/usr/bin/env python3
import sys
import os

# Ensure we can import xplane_pb2 from the same directory
sys.path.append(os.path.dirname(__file__))

try:
    import xplane_pb2
except ImportError:
    print("Error: Could not import xplane_pb2.")
    sys.exit(1)

def inspect(filepath):
    print(f"Inspecting {filepath}...")
    try:
        with open(filepath, 'rb') as f:
            data = f.read()
        
        xspace = xplane_pb2.XSpace()
        xspace.ParseFromString(data)
        
        print(f"Successfully parsed XSpace. Size: {len(data)} bytes.")
        print(f"Hostnames: {list(xspace.hostnames)}")
        print(f"Planes: {len(xspace.planes)}")
        
        max_lines = 3
        max_events = 5

        for i, plane in enumerate(xspace.planes):
            print(f"Plane {i}: ID={plane.id}, Name='{plane.name}'")
            print(f"  Lines: {len(plane.lines)}")
            print(f"  Event Metadata count: {len(plane.event_metadata)}")
            print(f"  Stat Metadata count: {len(plane.stat_metadata)}")
            
            # Check for event metadata integrity (sample)
            if plane.event_metadata:
                first_id = next(iter(plane.event_metadata))
                print(f"  Sample Metadata ID: {first_id} -> Name: {plane.event_metadata[first_id].name}")

            for line_idx, line in enumerate(plane.lines[:max_lines]):
                display_id_str = f", DisplayID={line.display_id}" if line.display_id != 0 else ""
                print(f"    Line {line_idx}: ID={line.id}{display_id_str}, Name='{line.name}', Events={len(line.events)}")
                base_ts = line.timestamp_ns

                for event_idx, event in enumerate(line.events[:max_events]):
                    offset_ns = event.offset_ps / 1000.0 if event.HasField("offset_ps") else 0.0
                    event_ts = base_ts + offset_ns
                    meta_name = plane.event_metadata.get(event.metadata_id).name if event.metadata_id in plane.event_metadata else "<unknown>"
                    print(
                        f"      Event {event_idx}: ts={event_ts:.0f} ns, duration={event.duration_ps} ps, meta='{meta_name}'"
                    )

                if len(line.events) > max_events:
                    print(f"      ... {len(line.events) - max_events} more events ...")

            if len(plane.lines) > max_lines:
                print(f"    ... {len(plane.lines) - max_lines} more lines ...")

    except Exception as e:
        print(f"FAILED to parse: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: ./inspect_trace.py <trace.pb>")
        sys.exit(1)
    inspect(sys.argv[1])

