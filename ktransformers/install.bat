@echo off

REM clear build dirs
rmdir /S /Q ktransformers\ktransformers_ext\build
rmdir /S /Q ktransformers\ktransformers_ext\cuda\build
rmdir /S /Q ktransformers\ktransformers_ext\cuda\dist
rmdir /S /Q ktransformers\ktransformers_ext\out
del /F /Q ktransformers\ktransformers_ext\cuda\*.egg-info

echo Installing python dependencies from requirements.txt
pip install -r requirements-local_chat.txt

echo Installing ktransformers
set KTRANSFORMERS_FORCE_BUILD=TRUE
pip install . --no-build-isolation
echo Installation completed successfully