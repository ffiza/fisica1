# Animaciones de problemas de $N$-cuerpos

Ejemplos de simulaciones de sistemas de $N$-cuerpos mediante gravedad newtoniana.

Los archivos de configuración relacionados con la física (paso temporal para la integración, longitud de suavizado, constante gravitatoria) para cada simulación pueden encontrarse en `configs/` en formato `yml`, las condiciones iniciales están en `ics/`, los resultados de cada corrida son almacenados en `results/` y las configuraciones globales (comunes a todas las corridas) se encuentran en `configs/global.yml`.

Algunas animaciones pueden encontrarse en [YouTube](https://www.youtube.com/fgiza/videos).

Para animar los resultados de una simulación dada (por ejemplo, `simulation_p01_ic01`) utilizando PyGame, correr:

```
python nbody/animation.py --result simulation_p01_ic01
```

Para correr tus propias simulaciones, podés crear un nuevo archivo de configuarción en el directorio `configs/` y un archivo de condiciones iniciales en `ics/` y luego ejecutar la simulación y/o animación:

```
python nbody/nbody.py --physics my_physics_file_name_suffix --ic my_ic_file_name_suffix
```

### Algunos ejemplos

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
