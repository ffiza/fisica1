# Animaciones de problemas de $N$-cuerpos

Ejemplos de simulaciones de sistemas de $N$-cuerpos mediante gravedad newtoniana. Los archivos de configuración para cada simulación pueden encontrarse en `animations/nbody/configs/` en formato `yml`; los resultados de cada corrida son almacenados en `animations/nbody/data/`. Cada simulación tiene un nombre asignado, según se describe a continuación.

Algunas animaciones pueden encontrarse en [YouTube](https://www.youtube.com/fgiza/videos).

Para generar los resultados de una simulación dada (por ejemplo, `two_body_example1`) utilizando PyGame, correr:

```
python nbody/animation.py --simulation two_body_example1
```

Para generar una película (actualmente, estoy sólo funciona en Ubuntu), utilizar `save_frames: true` en el archivo de configuración, y, luego de correr la simulación, ejecutar:

```
python movie.py --simulation two_body_example1 --delete_frames yes
```

El último argumento borra todos los frames una vez creada la película.

Para correr tus propias simulaciones, podés crear un nuevo archivo de configuarción en el directorio `configs/` y luego ejecutar la simulación y/o animación.

### Algunos ejemplos

#### [Ejemplo 1](https://youtu.be/I3U7MGbQIdA): Órbitas parabólicas (`two_body_example1`)

Un ejemplo de dos cuerpos que describen órbitas parabólicas (la energía mecánica total del sistema es nula).

#### [Ejemplo 2](https://youtu.be/8C-GpehjkiU): Órbitas parabólicas (`two_body_example2`)

Un ejemplo de dos cuerpos que describen órbitas parabólicas, pero la mínima distancia entre ambos en la mitad de la distancia inicial.

#### [Ejemplo 3](https://youtu.be/itIvMWKCWQ0): Órbitas elípticas (`two_body_example3`)

Un ejemplo de dos cuerpos con energía mecánica total negativa que describen órbitas elípticas.

#### [Ejemplo 4](https://youtu.be/DUorm2F3x1w): Órbitas elípticas con errores numéricos (`two_body_example4`)

Un ejemplo de dos cuerpos con energía mecánica total negativa que describen órbitas elípticas claramente distorsionadas debido a errores numéricos en la integración de las ecuaciones diferenciales.

#### [Ejemplo 5](https://youtu.be/Qin4mXVgOFM): Órbitas elípticas con mayor resolución temporal (`two_body_example5`)

El caso anterior, pero con un paso temporal menor para prevenir errores numéricos en la integración.
