import os
t = r'F:\Codebase\images_to_play\test_data\tandt_db\tandt\truck'
for folder in ['preset_tests', 'enhanced_tests', 'mvs_tests']:
    d = os.path.join(t, folder)
    if os.path.exists(d):
        for f in sorted(os.listdir(d)):
            if f.endswith('.ply'):
                s = os.path.getsize(os.path.join(d, f)) / (1024*1024)
                print(f'  {folder}/{f}: {s:.1f} MB')
