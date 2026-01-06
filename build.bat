uv run pyinstaller ./src/main.py ^
    --name anonify ^
    --onefile ^
    --noconsole ^
    --icon .\assets\icon.ico ^
    --noupx ^
    --add-data .\models\yolo11n.pt:.\models