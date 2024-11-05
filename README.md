<div align="center">
    <h1>Animaciones de gravitación</h1>
</div>

<p align="center">
    <a href="https://www.python.org/"><img src="https://forthebadge.com/images/badges/made-with-python.svg"></a>
</p>

Códigos de Python para resolver y animar la dinámica de masas que interactúan mediante fuerzas gravitatorias. El script principal es `nbody/nobdy.py`, que simula la evolución de un sistema de masas usando un integrador temporal tipo *leapfrog*. El script `nbody/animation.py` puede utilizarse para generar una animación de los resultados obtenidos utilizando PyGame. Todas las unidades están en el sistema internacional.

Para correr estos códigos es necesario instalar las siguientes librerías: NumPy, PyGame, pandas, tqdm y PyYAML, lo cual puede hacerse mediante el comando:

```bash
pip install numpy pygame-ce pandas tqdm pyyaml
```

<p align="center">
    <a href="https://i.imgur.com/rdGvizO.png"><img src="https://i.imgur.com/rdGvizO.png" width=300></a>
    <a href="https://i.imgur.com/IDAVO4n.png"><img src="https://i.imgur.com/IDAVO4n.png" width=300></a>
    <a href="https://i.imgur.com/LnKM7cf.png"><img src="https://i.imgur.com/LnKM7cf.png" width=300></a>
    <a href="https://i.imgur.com/eYiF0Dx.png"><img src="https://i.imgur.com/eYiF0Dx.png" width=300></a>
</p>

# Clonar o descargar este repositorio

Para tener una copia de este repositorio en tu sistema, podés clonar el mismo usando

```bash
git clone https://github.com/ffiza/nbody.git
```

o bien descargar el repositorio comprimido en formato ZIP yendo a `Code > Download ZIP`.

# Cómo animar las simulaciones incluidas

Para animar las simulaciones inlcuidas en este repositorio (cuyos resultados están almacenados en el directorio `results/`, en una archivo CSV por simulación) sólo es necesario ejecutar

```bash
python nbody/animation.py --result [nombre_resultados]
```

Por ejemplo, para animar los resultados dentro del archivo `simulation_p01_ic01.csv`, ejecutar

```bash
python nbody/animation.py --result simulation_p01_ic01
```

La simulación inicia pausada y puede activarse presionando `P`. Además, `R` resetea la animación y `D` muestra o esconde el panel del debugger.

# Cómo ejecutar tus propias simulaciones

Para ejecutar tus propias simulaciones, es necesario que crees un nuevo archivo dentro del directorio `configs/` y otro dentro del directorio `ics/`, como se describe más abajo. Luego, simplemente podés correr

```bash
python nbody/nbody.py --physics [nombre_archivo_configs] --ic [nombre_archivo_ics]
```

Python va a correr toda la simulación (tené en cuenta que cuantas más partículas haya más lenta va a ser la ejecución) y guardar los resultados en el directorio `results/`, en un archivo CSV.

Finalmente, para animar tus resultados, podés correr

```bash
python nbody/animation.py --result [nombre_resultados]
```

# Sobre los archivos de configuración

Los archivos de configuración relacionados con la física (paso temporal para la integración, longitud de suavizado, constante gravitatoria) para cada simulación pueden encontrarse en `configs/` en formato `yml`, las condiciones iniciales están en `ics/`, los resultados de cada corrida son almacenados en `results/` y las configuraciones globales (comunes a todas las corridas) se encuentran en `configs/global.yml`.

### Algunos ejemplos

Algunas animaciones pueden encontrarse en [YouTube](https://www.youtube.com/fgiza/videos).

#### [Ejemplo 1](https://youtu.be/I3U7MGbQIdA): Órbitas parabólicas (`simulation_p01_ic01`)

Un ejemplo de dos cuerpos que describen órbitas parabólicas (la energía mecánica total del sistema es nula).

#### [Ejemplo 2](https://youtu.be/8C-GpehjkiU): Órbitas parabólicas (`simulation_p02_ic02`)

Un ejemplo de dos cuerpos que describen órbitas parabólicas, pero la mínima distancia entre ambos en la mitad de la distancia inicial.

#### [Ejemplo 3](https://youtu.be/itIvMWKCWQ0): Órbitas elípticas (`simulation_p03_ic03`)

Un ejemplo de dos cuerpos con energía mecánica total negativa que describen órbitas elípticas.

#### [Ejemplo 4](https://youtu.be/DUorm2F3x1w): Órbitas elípticas con errores numéricos (`simulation_p04_ic04`)

Un ejemplo de dos cuerpos con energía mecánica total negativa que describen órbitas elípticas claramente distorsionadas debido a errores numéricos en la integración de las ecuaciones diferenciales.

#### [Ejemplo 5](https://youtu.be/Qin4mXVgOFM): Órbitas elípticas con mayor resolución temporal (`simulation_p05_ic05`)

El caso anterior, pero con un paso temporal menor para prevenir errores numéricos en la integración.
