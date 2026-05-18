# restart-server

Kill any running Flask server and restart it cleanly.

## Steps

1. Kill any process on port 5000:
```bash
# Windows
for /f "tokens=5" %a in ('netstat -aon ^| findstr :5000') do taskkill /F /PID %a
```

Or use PowerShell:
```powershell
Get-Process -Id (Get-NetTCPConnection -LocalPort 5000 -ErrorAction SilentlyContinue).OwningProcess | Stop-Process -Force
```

2. Start the server:
```bash
cd simple_splat/App
python app.py
```

Server runs at http://localhost:5000. It will print `ML-Sharp available: True/False` on startup.

## Notes
- `use_reloader=False` is intentional — reloader breaks ML-Sharp detection
- COLMAP DLLs are loaded from `C:\COLMAP\bin` (added to PATH in app.py)
- If ML-Sharp shows False, check that `sharp --help` works in terminal
