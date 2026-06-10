import os

p = r'F:\Codebase\images_to_play\simple_splat\App\gsplat_mcmc_trainer.py'
with open(p, 'r') as f:
    lines = f.readlines()

changes = 0

for i, line in enumerate(lines):
    # Fix comment on line 78
    if 'gsplat viewmats expect w2c' in line:
        lines[i] = line.replace('gsplat viewmats expect w2c as 4x4, so pad the 3x4 matrix',
                                'gsplat viewmats expect camera-to-world (c2w), so invert it')
        changes += 1
        print(f'Fixed line {i+1}: comment')
    
    # After w2c_4x4[:3, :] = w2c_3x4, add inversion line
    if 'w2c_4x4[:3, :] = w2c_3x4' in line:
        # Check if next line is already the inversion
        if i+1 < len(lines) and 'c2w_4x4 = np.linalg.inv' in lines[i+1]:
            print('Already fixed!')
        else:
            indent = line[:len(line) - len(line.lstrip())]
            lines.insert(i+1, indent + 'c2w_4x4 = np.linalg.inv(w2c_4x4)\n')
            changes += 1
            print(f'Fixed line {i+2}: added c2w inversion')
    
    # Change c2w_mats.append(torch.from_numpy(w2c_4x4)) to use c2w_4x4
    if 'c2w_mats.append(torch.from_numpy(w2c_4x4))' in line:
        lines[i] = line.replace('torch.from_numpy(w2c_4x4)', 'torch.from_numpy(c2w_4x4)')
        changes += 1
        print(f'Fixed line {i+1}: use c2w_4x4')

with open(p, 'w') as f:
    f.writelines(lines)

print(f'\nTotal changes: {changes}')
