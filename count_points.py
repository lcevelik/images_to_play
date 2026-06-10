import struct
with open(r'F:\Codebase\images_to_play\test_data\tandt_db\tandt\truck\sparse\0\points3D.bin', 'rb') as f:
    data = f.read()
count = 0
pos = 0
while pos < len(data):
    if pos + 8 > len(data):
        break
    pid = struct.unpack('q', data[pos:pos+8])[0]
    pos += 8 + 24 + 3 + 8  # id + xyz + rgb + error
    track_len = struct.unpack('Q', data[pos:pos+8])[0]
    pos += 8 + track_len * 8
    count += 1
print(f'COLMAP sparse points: {count:,}')
