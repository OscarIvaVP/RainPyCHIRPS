import subprocess

# Lista de scripts en el orden en que deben ejecutarse
scripts = [
    "1_creacion_carpetas.py",
    "2_descargar_chirps.py",
    "3_descomprimir.py",
    "4_pp_chirps_mes_areas.py",
    "5_analisis_espacial.py",
    "6_analisis_temporal.py",
    "7_prediccion.py"
]

for script in scripts:
    print(f"Ejecutando: {script}")
    result = subprocess.run(["python", script], capture_output=True, text=True)
    if result.returncode == 0:
        print(f"{script} ejecutado con éxito.")
    else:
        print(f"Error ejecutando {script}:")
        print(result.stderr)
