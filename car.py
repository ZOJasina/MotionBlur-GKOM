import glm
import glfw

class Car:
    """Represents the state of the car - position, velocity, orientation. Implements physics logic."""

    def __init__(self, initial_position=glm.vec3(0.0, -0.35, 0.0)):
        # constants
        self.ACCELERATION = 0.75
        self.FRICTION = 0.5
        self.MAX_SPEED = 3.0
        self.MAX_SPEED_REVERSE = -1.0
        self.ROTATION_SPEED = 0.5

        # state variables
        self.position = initial_position
        self.prev_position = glm.vec3(initial_position)
        self.speed = 0.0            # units per second
        self.angle = 0.0            # radians (0.0 -> along the -Z)

    def update(self, window, delta_time):
        """Updates car state."""

        self.prev_position = glm.vec3(self.position)

        is_accelerating = glfw.get_key(window, glfw.KEY_W) == glfw.PRESS
        is_braking = glfw.get_key(window, glfw.KEY_S) == glfw.PRESS

        if is_accelerating:
            self.speed += self.ACCELERATION * delta_time
            self.speed = min(self.speed, self.MAX_SPEED)
        elif is_braking:
            self.speed -= self.ACCELERATION * delta_time
            self.speed = max(self.speed, self.MAX_SPEED_REVERSE)
        else:
            if abs(self.speed) > 0:
                friction_loss = self.FRICTION * delta_time

                if self.speed > 0:
                    self.speed -= friction_loss
                    self.speed = max(0.0, self.speed)
                elif self.speed < 0:
                    self.speed += friction_loss
                    self.speed = min(0.0, self.speed)

        is_turning_right = glfw.get_key(window, glfw.KEY_D) == glfw.PRESS
        is_turning_left = glfw.get_key(window, glfw.KEY_A) == glfw.PRESS

        rotation_modifier = 1.0 if self.speed >= 0 else -1.0
        if is_turning_right:
            self.angle -= self.ROTATION_SPEED * delta_time * rotation_modifier
        elif is_turning_left:
            self.angle += self.ROTATION_SPEED * delta_time * rotation_modifier

        direction_x = -glm.sin(self.angle)   # X (left/right)
        direction_z = -glm.cos(self.angle)  # Z (forward/backward)
        direction = glm.vec3(direction_x, 0.0, direction_z)
        self.position += direction * self.speed * delta_time

    def get_model_matrix(self):
        """Returns the model's matrix based on current position and orientation."""
        model = glm.mat4(1.0)
        model = glm.translate(model, self.position)
        model = glm.rotate(model, self.angle, glm.vec3(0.0, 1.0, 0.0))
        model = glm.rotate(model, glm.radians(180.0), glm.vec3(0.0, 1.0, 0.0))
        model = glm.scale(model, glm.vec3(0.2))

        return model