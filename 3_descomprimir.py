import gzip
import os

# Define la carpeta que contiene los archivos .gz
input_directory = "output/imagenes-chirps-mes/comprimidos"
output_directory = os.path.join("output/imagenes-chirps-mes")

# Crea la carpeta de salida si no existe
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Recorre todos los archivos en la carpeta
for file_name in os.listdir(input_directory):
    if file_name.endswith(".gz"):
        input_file_path = os.path.join(input_directory, file_name)
        output_file_path = os.path.join(output_directory, file_name[:-3])  # Elimina la extensión .gz

        try:
            # Descomprime el archivo
            with gzip.open(input_file_path, 'rb') as gz_file:
                with open(output_file_path, 'wb') as out_file:
                    out_file.write(gz_file.read())
            print(f"Archivo {file_name} descomprimido como {os.path.basename(output_file_path)}")
        except EOFError:
            print(f"Error: El archivo {file_name} está corrupto. Eliminando y volviendo a intentar...")
            os.remove(input_file_path)

print("Proceso de descompresión completado.")
