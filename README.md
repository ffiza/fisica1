# Física 1

Scripts escritos en Python para generar animaciones para la materia Física 1 de la Facultad de Ciencias Exactas y Naturales de la Universidad de Buenos Aires.

Algunas animaciones pueden encontrarse en [YouTube](https://www.youtube.com/fgiza/videos).

## Problema de $N$-cuerpos

### [Ejemplo 1](https://youtu.be/I3U7MGbQIdA): Órbitas parabólicas

Un ejemplo de dos cuerpos que describen órbitas parabólicas (la energía mecánica total del sistema es nula).

Para reproducir esta animación, correr:

```python animations/nbody.py --masses 6.0 6.0 --xpositions 3.0 -3.0 --ypositions 0.0 0.0 --xvelocities -1.0816 1.0816 --yvelocities 2.9716 -2.9716 --timestep 0.001 --steps 15000 --fps 50 --one_every 20 --filename "two-body-example1"```

### [Ejemplo 2](https://youtu.be/8C-GpehjkiU): Órbitas parabólicas

Un ejemplo de dos cuerpos que describen órbitas parabólicas, pero la mínima distancia entre ambos en la mitad de la distancia inicial.

Para reproducir esta animación, correr:

```python animations/nbody.py --masses 6.0 6.0 --xpositions 3.0 -3.0 --ypositions 0.0 0.0 --xvelocities -2.2361 2.2361 --yvelocities -2.2361 2.2361 --timestep 0.001 --steps 15000 --fps 50 --one_every 20 --filename "two-body-example2"```

### [Ejemplo 3](https://youtu.be/itIvMWKCWQ0): Órbitas elípticas


Un ejemplo de dos cuerpos con energía mecánica total negativa que describen órbitas elípticas.

Para reproducir esta animación, correr:

```python animations/nbody.py --masses 6.0 6.0 --xpositions 3.0 -3.0 --ypositions 0.0 0.0 --xvelocities -0.6840 0.6840 --yvelocities 1.8794 -1.8794 --timestep 0.001 --steps 15000 --fps 50 --one_every 20 --filename "two-body-example3"```

### [Ejemplo 4](https://youtu.be/DUorm2F3x1w): Órbitas elípticas con errores numéricos

Un ejemplo de dos cuerpos con energía mecánica total negativa que describen órbitas elípticas claramente distorsionadas debido a errores nuḿericos en la integración de las ecuaciones diferenciales.

Para reproducir esta animación, correr:

```python animations/nbody.py --masses 6.0 6.0 --xpositions 3.0 -3.0 --ypositions 0.0 0.0 --xvelocities 0.7071 -0.7071 --yvelocities 0.7071 -0.7071 --timestep 0.01 --steps 1500 --fps 50 --one_every 2 --filename "two-body-example4"```

### [Ejemplo 5](https://youtu.be/Qin4mXVgOFM): Órbitas elípticas con mayor resolución temporal

El caso anterior, pero con un paso temporal menor para prevenir errores numéricos en la integración.

Para reproducir esta animación, correr:

```python animations/nbody.py --masses 6.0 6.0 --xpositions 3.0 -3.0 --ypositions 0.0 0.0 --xvelocities 0.7071 -0.7071 --yvelocities 0.7071 -0.7071 --timestep 0.001 --steps 15000 --fps 50 --one_every 20 --filename "two-body-example5"```

### Ejemplo 6: Cuatro cuerpos con errores numéricos

Un ejemplo de cuatro partículas que orbitan partiendo de una condición inicial simétrica hasta que errores numéricos inducen perturbaciones en las órbitas.

```python animations/nbody.py --masses 1.0 1.0 1.0 1.0 --xpositions -2.0 0.0 2.0 0.0 --ypositions 0.0 2.0 0.0 -2.0 --xvelocities 0.0 1.0 0.0 -1.0 --yvelocities 1.0 0.0 -1.0 0.0 --timestep 0.001 --steps 20000 --fps 50 --one_every 20 --filename "four-body-example1"```