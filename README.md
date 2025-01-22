# RainPYCHIRPS

**RainPYCHIRPS** es un software de código abierto desarrollado en Python para el procesamiento y análisis automatizado de datos de precipitación satelital CHIRPS. Este repositorio incluye herramientas para descargar, descomprimir, analizar y visualizar datos, proporcionando salidas en gráficos, tablas y documentos Word, ideales para estudios hidrológicos y climáticos.

## Características principales

- **Automatización Completa**: Scripts secuenciales que automatizan todo el proceso, desde la descarga de datos hasta la generación de informes.
- **Procesamiento de Datos Espaciales**: Análisis espacial utilizando shapefiles y datos CHIRPS.
- **Análisis Temporal**: Análisis y visualización de patrones temporales de precipitación.
- **Predicción de Precipitación**: Modelos de aprendizaje automático para predicciones futuras con intervalos de confianza.
- **Generación de Informes**: Exportación de resultados en tablas (Excel), gráficos y documentos Word.

## Estructura del Repositorio

```plaintext
RainPYCHIRPS/
├── input/                 # Carpeta para datos de entrada ()
├── output/                # Resultados generados (gráficos, tablas, informes)
│   ├── datos-tabulares/   # Resultados tabulares en formato Excel
│   ├── graficos/          # Gráficos generados
│   └── informes/          # Informes en Word
├── 1_creacion_carpetas.py # Script para crear estructura de carpetas
├── 2_descargar_chirps.py  # Script para descargar datos CHIRPS
├── 3_descomprimir.py      # Script para descomprimir archivos CHIRPS
├── 4_pp_chirps_mes_areas.py # Procesamiento y análisis espacial de datos CHIRPS
├── 5_analisis_espacial.py # Análisis espacial y generación de gráficos
├── 6_analisis_temporal.py # Análisis temporal y generación de gráficos
├── 7_prediccion.py        # Predicción de precipitación con modelos
├── main.py                # Script principal para ejecutar todo el flujo
├── requirements.txt       # Dependencias necesarias
└── README.md              # Documentación del repositorio
```

## Instalación

1. Clona este repositorio:
   ```bash
   git clone https://github.com/tu-usuario/RainPYCHIRPS.git
   cd RainPYCHIRPS
   ```

2. Crea un entorno virtual (opcional, pero recomendado):
   ```bash
   python -m venv env
   source env/bin/activate  # En Windows: env\Scripts\activate
   ```

3. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```

## Uso

1. **Configura los datos de entrada**: Coloca los shapefiles en la carpeta `input/area-estudio/`.
2. **Ejecuta el script principal**:
   ```bash
   python main.py
   ```
3. Los resultados estarán disponibles en la carpeta `output/`.

## Scripts Principales

- **1_creacion_carpetas.py**: Crea la estructura de carpetas necesaria.
- **2_descargar_chirps.py**: Descarga datos CHIRPS desde el 1981 hasta el año actual.
- **3_descomprimir.py**: Descomprime archivos CHIRPS (.tif.gz).
- **4_pp_chirps_mes_areas.py**: Procesa datos CHIRPS para calcular promedios de precipitación en áreas específicas.
- **5_analisis_espacial.py**: Genera gráficos espaciales con shapefiles y datos CHIRPS procesados.
- **6_analisis_temporal.py**: Analiza patrones temporales y tendencias de precipitación.
- **7_prediccion.py**: Predice precipitación futura utilizando modelos de aprendizaje automático.
- **main.py**: Ejecuta todos los scripts en el orden correcto.

## Requisitos del Sistema

- Python 3.8 o superior
- Conexión a Internet para descargar datos CHIRPS

## Contribuciones

¡Contribuciones son bienvenidas! Si deseas mejorar este proyecto, por favor realiza un fork del repositorio y envía un pull request.

## Licencia

Este proyecto está licenciado bajo la Licencia MIT. Consulta el archivo `LICENSE` para más detalles.

## Contacto

Para consultas o comentarios, puedes contactarme en [oscarvargas@trabajocientific.com].

---

¡Explora y analiza datos de precipitación con **RainPYCHIRPS**!
