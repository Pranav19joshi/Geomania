@echo off
echo Starting Geomania Backend Server...
echo The application will be available at: http://127.0.0.1:8000
echo Press Ctrl+C to stop the server.
echo.

.venv\Scripts\uvicorn.exe app.main:app
