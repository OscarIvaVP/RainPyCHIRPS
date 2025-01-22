import requests
import os

# Define el directorio donde se guardarán los archivos
output_directory = os.path.join(os.getcwd(), "output/imagenes-chirps-mes/comprimidos")

# Crea el directorio si no existe
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Archivo de registro para controlar el progreso
download_log = "seguimiento_descarga_chirps.txt"

# Cargar el progreso previo desde el archivo de registro
if os.path.exists(download_log):
    with open(download_log, 'r') as log_file:
        downloaded_files = set(log_file.read().splitlines())
else:
    downloaded_files = set()

# Crear la lista de archivos a descargar
chirps_files = []
for year in range(1981, 2025):
    for month in range(1, 13):
        file_name = f"chirps-v2.0.{year}.{month:02d}.tif.gz"
        url = f"https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_monthly/tifs/{file_name}"
        chirps_files.append(url)

# Descargar los archivos restantes
for file_url in chirps_files:
    file_name = os.path.basename(file_url)

    # Saltar archivos ya descargados
    if file_name in downloaded_files:
        print(f"Archivo {file_name} ya registrado como descargado. Omitiendo...")
        continue

    # Ruta completa del archivo
    file_path = os.path.join(output_directory, file_name)

    # Descargar archivo
    try:
        print(f"Descargando {file_name}...")
        response = requests.get(file_url, stream=True, timeout=30)
        if response.status_code == 200:
            with open(file_path, 'wb') as file:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        file.write(chunk)
            print(f"Archivo {file_name} descargado con éxito.")
            # Registrar el archivo como descargado
            with open(download_log, 'a') as log_file:
                log_file.write(f"{file_name}\n")
        else:
            print(f"Error al descargar {file_name}: Código {response.status_code}")
    except Exception as e:
        print(f"Error al descargar {file_name}: {e}")
