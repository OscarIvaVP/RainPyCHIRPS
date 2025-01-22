import os
import shutil

# Crear las carpetas necesarias si no existen
if not os.path.exists("output/datos-tabulares"):
    os.makedirs("output/datos-tabulares")

if not os.path.exists("output/imagenes-chirps-mes/comprimidos"):
    os.makedirs("output/imagenes-chirps-mes/comprimidos")

if not os.path.exists("output/graficos"):
    os.makedirs("output/graficos")

if not os.path.exists("output/informes"):
    os.makedirs("output/informes")

if not os.path.exists("input"):
    os.makedirs("input")

# Verificar si la carpeta "area-estudio" existe en el directorio de trabajo
if os.path.exists("area-estudio") and os.path.isdir("area-estudio"):
    # Mover la carpeta "area-estudio" dentro de "input"
    shutil.move("area-estudio", "input/area-estudio")
    print("La carpeta 'area-estudio' se ha movido dentro de 'input'.")
else:
    print("La carpeta 'area-estudio' no existe en el directorio de trabajo.")
