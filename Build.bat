@echo off
set SDL2_PATH=external\sdl2
set IMGUI_PATH=external\imgui
set IMGPLOT_PATH=external\implot
set OPENGL_PATH=external\opengl

rem Check for release or debug argument
if "%1"=="release" (
    set BUILD_TYPE=Release
    set COMPILER_OPTS=/O2 /MT /EHsc /W3 /std:c++20 /DNDEBUG
    set OUTPUT_DIR=Build\Release
) else (
    set BUILD_TYPE=Debug
    set COMPILER_OPTS=/EHsc /MT /Od /W3 /std:c++20 /Zi
    set OUTPUT_DIR=Build\Debug
)

rem Set CUDA path
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"

rem Include directories
set INCLUDE_OPTS=/I %SDL2_PATH%\include /I %OPENGL_PATH% /I %IMGUI_PATH%\include /I %IMGUI_PATH%\backends /I %IMGPLOT_PATH%\include /I "%CUDA_PATH%\include"

rem Libraries to link
set LIB_OPTS=/link /LIBPATH:%SDL2_PATH%\lib /LIBPATH:"%CUDA_PATH%\lib\x64" SDL2.lib opengl32.lib user32.lib gdi32.lib shell32.lib cudart.lib

rem Create Build folder if it doesn't exist
if not exist Build mkdir Build
if not exist %OUTPUT_DIR% mkdir %OUTPUT_DIR%

rem Clean object files for release builds
if "%BUILD_TYPE%"=="release" (
    del /Q %OUTPUT_DIR%\*.obj
)

rem Compile CUDA file
echo Compiling CUDA file...
nvcc -c Src\VerletCuda.cu -o %OUTPUT_DIR%\VerletCuda.obj -I "%CUDA_PATH%\include" -L "%CUDA_PATH%\lib\x64" -lcudart
if errorlevel 1 (
    echo CUDA compilation failed
    exit /b 1
)

rem Compile and link Engine.cpp into object files and output to build folder
echo Compiling Engine.cpp...
cl /Fo:%OUTPUT_DIR%\ /Fe:%OUTPUT_DIR%\engine.exe /Fd:%OUTPUT_DIR%\engine.pdb %COMPILER_OPTS% %INCLUDE_OPTS% Src\Engine.cpp %OUTPUT_DIR%\VerletCuda.obj %LIB_OPTS%
if errorlevel 1 (
    echo Compilation failed
    exit /b 1
)

rem Copy Assets folder to output directory
xcopy /E /I /Y Assets %OUTPUT_DIR%\Assets

rem Copy SDL2.dll to output directory
copy /Y %SDL2_PATH%\lib\SDL2.dll %OUTPUT_DIR%\
