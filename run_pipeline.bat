@echo off
echo Running Preprocessing...
python preprocess.py
echo.
echo Running Model Training...
python train_model.py
echo.
echo Done! Please run python predict.py your_audio.wav to test.
pause
