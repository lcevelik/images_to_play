import os
p = r'F:\Codebase\images_to_play\simple_splat\App\gsplat_mcmc_trainer.py'
with open(p, 'r') as f:
    c = f.read()
if 'rgb_tensor / C0' in c:
    c = c.replace('rgb_tensor / C0', '(rgb_tensor - 0.5) / C0')
    with open(p, 'w') as f:
        f.write(c)
    print('Fixed! Restored (rgb - 0.5) / C0')
elif '(rgb_tensor - 0.5) / C0' in c:
    print('Already correct!')
else:
    print('Unknown state')
    for i, line in enumerate(c.split('\n'), 1):
        if 'C0' in line or 'sh0' in line:
            print(f'{i}: {line.strip()}')
