import struct

path = r'F:\Codebase\images_to_play\test_data\tandt_db\tandt\truck\sparse\0\points3D.bin'
with open(path, 'rb') as f:
    data = f.read()

print(f"File size: {len(data):,} bytes")

# First 8 bytes = num_points (uint64)
num_points = struct.unpack('Q', data[0:8])[0]
print(f"Header says: {num_points:,} points")

# Skip header, start parsing
pos = 8
count = 0
while pos < len(data) and count < 5:
    xyz = struct.unpack('ddd', data[pos:pos+24])
    rgb = struct.unpack('BBB', data[pos+24:pos+27])
    error = struct.unpack('d', data[pos+27:pos+35])
    track_len = struct.unpack('Q', data[pos+35:pos+43])
    
    print(f"  Point {count}: xyz=({xyz[0]:.1f},{xyz[1]:.1f},{xyz[2]:.1f}), track_len={track_len}")
    
    pos = pos + 43 + track_len * 8
    count += 1

print(f"\nExpected points from header: {num_points:,}")

# Quick estimate from file size
# avg track ~5 views per point: 43 + 5*8 = 83 bytes per point
est_points = (len(data) - 8) / 83
print(f"Estimated from file size (~5 views/pt): ~{int(est_points):,}")
