import struct
import sys

path = r'F:\Codebase\images_to_play\test_data\tandt_db\tandt\truck\sparse\0\points3D.bin'
with open(path, 'rb') as f:
    data = f.read()

print(f"File size: {len(data):,} bytes")

count = 0
pos = 0
while pos < len(data):
    remaining = len(data) - pos
    if remaining < 43:
        print(f"Partial record at pos {pos}, remaining {remaining} bytes")
        break
    
    pid = struct.unpack('q', data[pos:pos+8])[0]
    xyz = struct.unpack('ddd', data[pos+8:pos+32])
    rgb = struct.unpack('BBB', data[pos+32:pos+35])
    error = struct.unpack('d', data[pos+35:pos+43])
    
    if pos + 43 + 8 > len(data):
        print(f"No room for track_len at pos {pos}")
        break
    
    track_len = struct.unpack('Q', data[pos+43:pos+51])[0]
    
    # Sanity check
    if track_len > 10000:
        print(f"Bad track_len={track_len} at pos {pos}, point {count}")
        break
    
    expected_end = pos + 43 + 8 + track_len * 8
    if expected_end > len(data):
        print(f"Track extends past EOF at point {count}: need {expected_end}, have {len(data)}")
        break
    
    if count < 5:
        print(f"  Point {count}: id={pid}, xyz=({xyz[0]:.2f},{xyz[1]:.2f},{xyz[2]:.2f}), track_len={track_len}")
    
    pos = expected_end
    count += 1

print(f"\nTotal COLMAP sparse points: {count:,}")
