cd ww2ogg/
for %%f in (../*.WEM) do ww2ogg.exe ../%%f --pcb packed_codebooks_aoTuV_603.bin
cd ../revorb/
for %%f in (../*.ogg) do revorb.exe ../%%f
pause
