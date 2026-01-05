uv run pyinstaller ./src/main.py ^
    --onefile ^
    --noupx ^
    --noconsole ^
    --name anonify ^
    --add-data .\models\yolo11n.pt:.\models