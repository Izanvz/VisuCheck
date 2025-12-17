# VisuCheck

VisuCheck es un proyecto personal de **análisis visual automatizado** que combina visión por computador, OCR y reglas de negocio para extraer y validar información a partir de imágenes. La idea principal es explorar cómo montar un pipeline completo y modular para tareas de verificación visual y análisis de productos.

No es un producto final ni una solución comercial cerrada. Es un proyecto técnico orientado a aprendizaje, experimentación y diseño de sistemas.

---

## Qué hace el proyecto

A alto nivel, VisuCheck permite:

- Analizar imágenes de estanterías o productos  
- Detectar regiones relevantes mediante modelos de visión por computador  
- Extraer texto con OCR  
- Interpretar ese texto usando reglas (precios, pesos, variantes, etc.)  
- Generar resultados estructurados y visualizables  

El foco está en **la arquitectura y el pipeline**, no en entrenar modelos desde cero ni en optimizar métricas.

---

## Cómo está planteado

El flujo general es el siguiente:

1. **Entrada**
   - Imágenes de ejemplo en `data/samples/`

2. **Detección**
   - Uso de YOLO para localizar zonas de interés

3. **Preprocesado**
   - Limpieza y adaptación de la imagen para mejorar el OCR

4. **OCR**
   - PaddleOCR como motor principal
   - Pensado para poder cambiar de OCR sin afectar al resto del sistema

5. **Reglas**
   - Parsing y validación del texto extraído
   - Reglas definidas en YAML para evitar acoplar lógica al código

6. **Salida**
   - Resultados exportables y visualización básica mediante dashboard

---

## Estructura del proyecto

```
VisuCheck/
├── configs/            # Configuración del sistema (datasets, OCR, reglas, etc.)
├── dashboard/          # Dashboard sencillo para visualizar resultados
├── data/
│   ├── samples/        # Imágenes de ejemplo versionadas
│   └── results/        # Resultados generados (no versionados)
├── scripts/            # Scripts para ejecutar distintas partes del pipeline
├── src/
│   ├── core/           # Lógica principal (detección, OCR, reglas)
│   ├── services/       # Orquestación del pipeline
│   ├── lang/           # Razonamiento y prompts
│   └── ui/             # Interfaz básica
├── tests/
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## Ejecución

Ejemplo básico para probar el pipeline con imágenes de muestra:

```bash
python -m scripts.analyze_samples
```

Existen otros scripts para análisis de estanterías, limpieza de precios, exportación de resultados y visualización.  
No hay un único entrypoint porque el proyecto está pensado para exploración y pruebas.

---

## Configuración

La mayor parte del comportamiento del sistema se controla desde `configs/` usando YAML:

- Configuración de datasets
- Parámetros del detector
- Ajustes de OCR
- Reglas de negocio (precios, pesos, variantes)

Esto permite cambiar lógica y comportamiento sin tocar el código base.

---

## Design Decisions

Esta sección resume algunas decisiones conscientes tomadas durante el desarrollo del proyecto.

### Pipeline modular
El sistema está dividido en detección, OCR y reglas para poder sustituir componentes sin romper el resto del pipeline. La prioridad fue mantener bajo acoplamiento y facilitar pruebas con distintas configuraciones.

### Uso de configuración externa (YAML)
Las reglas de negocio y gran parte de los parámetros viven fuera del código. Esto permite iterar rápido y modificar el comportamiento sin necesidad de reescribir lógica interna.

### Uso de modelos preentrenados
El proyecto no se centra en entrenar modelos propios, sino en integrar y orquestar componentes. Usar modelos preentrenados acelera la validación de la arquitectura y el flujo completo.

### Exclusión de datasets grandes del repositorio
Los datasets pesados (por ejemplo SKU110K) no se incluyen en Git para mantener el repositorio ligero y manejable. El proyecto asume ejecución local con datasets externos.

### Scripts independientes en lugar de una única app
Se optó por múltiples scripts ejecutables en lugar de una única aplicación cerrada para facilitar pruebas, depuración y exploración de distintas partes del sistema.

### Proyecto pausado de forma intencionada
El proyecto se deja en estado pausado una vez alcanzada una arquitectura clara y funcional. El objetivo principal fue aprender y diseñar una base reutilizable, no cerrar un producto final.

---

## Estado del proyecto

⏸ **Proyecto pausado**

El pipeline funciona y la estructura es extensible, pero no está orientado a producción ni despliegue. El foco ha sido el diseño del sistema y la experimentación.

---

## Notas

- Los datasets grandes no se incluyen en el repositorio.
- El entorno virtual no forma parte del control de versiones.
- El código está orientado a prototipado y aprendizaje.

---

## Licencia

Proyecto personal con fines educativos y experimentales.
