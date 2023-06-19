# Física 1

Useful scripts for animations, excersices, data analysis and presentations of the course *Física 1* of the Faculty of Exact and Natural Sciences of the Univeristy of Buenos Aires.

## Animations

### ```two-body-example1.mp4```

An example of two bodies describing a parabolic orbit (the total energy of the system is zero).

To reproduce this animation, run:

```python animations/nbody.py --masses 6.0 6.0 --xpositions 3.0 -3.0 --ypositions 0.0 0.0 --xvelocities -1.0816 1.0816 --yvelocities 2.9716 -2.9716 --timestep 0.001 --steps 15000 --fps 50 --one_every 20 --filename "two-body-example1"```

### ```two-body-example2.mp4```

An example of two bodies describing a parabolic orbit, but the minimum distance between the two is half the initial distance. Note that parabolic orbits are those with $E=0$ and are independent of the initial angular momentum of the system.

To reproduce this animation, run:

```python animations/nbody.py --masses 6.0 6.0 --xpositions 3.0 -3.0 --ypositions 0.0 0.0 --xvelocities -2.2361 2.2361 --yvelocities -2.2361 2.2361 --timestep 0.001 --steps 15000 --fps 50 --one_every 20 --filename "two-body-example2"```

### ```two-body-example3.mp4```

An example of two bodies with negative energy that describe an elliptical orbit.

To reproduce this animation, run:

```python animations/nbody.py --masses 6.0 6.0 --xpositions 3.0 -3.0 --ypositions 0.0 0.0 --xvelocities -0.6840 0.6840 --yvelocities 1.8794 -1.8794 --timestep 0.001 --steps 15000 --fps 50 --one_every 20 --filename "two-body-example3"```

### ```two-body-example4.mp4```

An example of two bodies with negative energy that describe an elliptical orbit that is clearly distorted due to numerical errors in the integration.

To reproduce this animation, run:

```python animations/nbody.py --masses 6.0 6.0 --xpositions 3.0 -3.0 --ypositions 0.0 0.0 --xvelocities 0.7071 -0.7071 --yvelocities 0.7071 -0.7071 --timestep 0.01 --steps 1500 --fps 50 --one_every 2 --filename "two-body-example4"```

### ```two-body-example5.mp4```

The same case as `two-body-example4.mp4` but with a smaller timestep to prevent numerical errors.

To reproduce this animation, run:

```python animations/nbody.py --masses 6.0 6.0 --xpositions 3.0 -3.0 --ypositions 0.0 0.0 --xvelocities 0.7071 -0.7071 --yvelocities 0.7071 -0.7071 --timestep 0.001 --steps 15000 --fps 50 --one_every 20 --filename "two-body-example5"```
