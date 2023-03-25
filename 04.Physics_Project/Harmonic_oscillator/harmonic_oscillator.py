import time

class Ball():
    def __init__(self, mass: float, position: float, velocity: float, acceleration: float) -> None:
        self.mass = mass
        self.position = position
        self.velocity = velocity
        self.acceleration = acceleration

def print_blanks(num: int) -> None:
    for i in range(int(num) + 39):
        print(" ", end="")
    return None

def main() -> None:
    spring_constant: float = 1. # Hooke's law: F = -kx
    damping_coeff: float = 0.1 # set to > 0 for damped harmonic oscillator
    time_step: float = 0.1
    ball = Ball(mass=0.1, position=-40., velocity=0., acceleration=0.)

    # Run as long the displacement (or velocity) are "significant"
    # Larger values will terminate the simulation earlier
    # Smaller values will terminate the simulation later on
    while (abs(ball.position) > 1.0) or (abs(ball.velocity) > 1.0):
        damping_force: float = -ball.velocity * damping_coeff # damping force is linearly dependent upon the velocity
        spring_force: float = -spring_constant * ball.position + damping_force  # the damping force term is to create a damped oscillator
                                                                                # the oscillation will be subjected to exponential decay 
                                                                                # which depends upon the damping coefficient

        ball.acceleration = spring_force / ball.mass # a = F / m
        ball.velocity += ball.acceleration * time_step # u = at
        ball.position += ball.velocity * time_step # x = ut

        print_blanks(ball.position)
        print("o")
        time.sleep(50_000 / 1_000_000.0)

    return None

if __name__ == "__main__":
    main()