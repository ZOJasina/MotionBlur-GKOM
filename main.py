import glfw
from OpenGL.GL import *
import glm
import ctypes
from vertex_models import *
from car import Car

def load_shader(shader_file, shader_type):
    """reads and compiles shader from file"""
    try:
        with open(shader_file, "r") as f:
            shader_source = f.read()
    except FileNotFoundError:
        print(f"Error: file not found {shader_file}")
        return None

    shader = glCreateShader(shader_type)
    glShaderSource(shader, shader_source)
    glCompileShader(shader)

    if not glGetShaderiv(shader, GL_COMPILE_STATUS):
        log = glGetShaderInfoLog(shader).decode("utf-8")
        print(f"Error compiling shader ({shader_file}):\n{log}")
        return None

    return shader


def create_shader_program(vertex_file, fragment_file):
    """Creates shader program"""
    vertex_shader = load_shader(vertex_file, GL_VERTEX_SHADER)
    fragment_shader = load_shader(fragment_file, GL_FRAGMENT_SHADER)

    if not vertex_shader or not fragment_shader:
        return None

    program = glCreateProgram()
    glAttachShader(program, vertex_shader)
    glAttachShader(program, fragment_shader)
    glLinkProgram(program)

    if not glGetProgramiv(program, GL_LINK_STATUS):
        log = glGetProgramInfoLog(program).decode("utf-8")
        print(f"Error linking program:\n{log}")
        return None

    glDeleteShader(vertex_shader)
    glDeleteShader(fragment_shader)

    return program

def setup_model(vertices_np):
    """Creates VAO + VBO for given vertex array."""
    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)

    vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, vertices_np.nbytes, vertices_np, GL_STATIC_DRAW)

    stride = 6 * vertices_np.itemsize
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(12))
    glEnableVertexAttribArray(1)

    glBindBuffer(GL_ARRAY_BUFFER, 0)
    glBindVertexArray(0)

    return vao, vbo


def render_model(shader, vao, vertex_count, model, view, projection):
    """Renders a model with given transformation matrices."""
    glUseProgram(shader)
    normalMatrix = glm.transpose(glm.inverse(glm.mat3(model)))

    # Set uniforms
    glUniformMatrix4fv(glGetUniformLocation(shader, "model"), 1, GL_FALSE, glm.value_ptr(model))
    glUniformMatrix4fv(glGetUniformLocation(shader, "view"), 1, GL_FALSE, glm.value_ptr(view))
    glUniformMatrix4fv(glGetUniformLocation(shader, "projection"), 1, GL_FALSE, glm.value_ptr(projection))
    glUniformMatrix3fv(glGetUniformLocation(shader, "normalMatrix"), 1, GL_FALSE, glm.value_ptr(normalMatrix))

    glBindVertexArray(vao)
    glDrawArrays(GL_TRIANGLES, 0, vertex_count)
    glBindVertexArray(0)
    return

def render_complex_model(shader, vao, draw_commands, model, view, projection):
    """Renders a complex (batched) model using draw commands."""
    glUseProgram(shader)

    # Set uniforms
    normalMatrix = glm.transpose(glm.inverse(glm.mat3(model)))
    glUniformMatrix4fv(glGetUniformLocation(shader, "model"), 1, GL_FALSE, glm.value_ptr(model))
    glUniformMatrix4fv(glGetUniformLocation(shader, "view"), 1, GL_FALSE, glm.value_ptr(view))
    glUniformMatrix4fv(glGetUniformLocation(shader, "projection"), 1, GL_FALSE, glm.value_ptr(projection))
    glUniformMatrix3fv(glGetUniformLocation(shader, "normalMatrix"), 1, GL_FALSE, glm.value_ptr(normalMatrix))

    glBindVertexArray(vao)


    for start_index, vertex_count in draw_commands:
        glDrawArrays(GL_TRIANGLES, start_index, vertex_count)

    glBindVertexArray(0)
    return

def render_motion_blur_model(shader, vao, draw_commands, model, view, projection,
                             velocity_world, samples=8, decay=0.75, base_alpha=0.9):
    """
    Object-space motion blur:
    - shader: program
    - vao, draw_commands: geometry
    - model, view, projection: glm mats
    - velocity_world: glm.vec3, world-space displacement per frame (current_pos - prev_pos)
    - samples: number of blur samples (>=1). Higher = smoother but slower.
    - decay: alpha decay per sample (0..1)
    - base_alpha: starting alpha for first sample
    """
    if samples <= 1 or glm.length(velocity_world) == 0.0:
        # No blur, just render normally
        render_complex_model(shader, vao, draw_commands, model, view, projection)
        return

    # Enable blending and keep depth test active; but disable depth writes so blur doesn't occlude
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glEnable(GL_DEPTH_TEST)
    glDepthMask(GL_FALSE)  # don't write depth for blur passes

    # Ensure shader active
    glUseProgram(shader)

    # Precompute uniform locations (could be cached for speed)
    loc_model = glGetUniformLocation(shader, "model")
    loc_view = glGetUniformLocation(shader, "view")
    loc_proj = glGetUniformLocation(shader, "projection")
    loc_normal = glGetUniformLocation(shader, "normalMatrix")
    loc_alpha = glGetUniformLocation(shader, "u_alpha")

    # For each sample, draw the object shifted slightly *backwards* along velocity
    # so the smear trails behind motion.
    for i in range(samples):
        t = (i + 1) / float(samples)  # from 1/samples .. 1.0
        offset = -velocity_world * t  # move backward along motion
        # build transformed model: translate by offset in world space, then apply original model
        # note: if your model matrix already contains translation, adding offset should be applied in world coords
        model_i = glm.translate(glm.mat4(1.0), glm.vec3(offset)) * model

        normalMatrix = glm.transpose(glm.inverse(glm.mat3(model_i)))

        if loc_model != -1: glUniformMatrix4fv(loc_model, 1, GL_FALSE, glm.value_ptr(model_i))
        if loc_view != -1: glUniformMatrix4fv(loc_view, 1, GL_FALSE, glm.value_ptr(view))
        if loc_proj != -1: glUniformMatrix4fv(loc_proj, 1, GL_FALSE, glm.value_ptr(projection))
        if loc_normal != -1: glUniformMatrix3fv(loc_normal, 1, GL_FALSE, glm.value_ptr(normalMatrix))

        # compute alpha for this sample
        alpha = base_alpha * (decay ** i)
        if loc_alpha != -1:
            glUniform1f(loc_alpha, float(alpha))

        glBindVertexArray(vao)
        for start_index, vcount in draw_commands:
            glDrawArrays(GL_TRIANGLES, start_index, vcount)
        glBindVertexArray(0)

    # restore state
    glDepthMask(GL_TRUE)
    glDisable(GL_BLEND)
    # note: keep depth test enabled (it was before)
    return

def load_model(path):
    vertices_np, draw_commands = load_model_batched(path)
    if vertices_np is None:
        print("Failed to load model, terminating.")
        glfw.terminate()
        return
    vao, vbo = setup_model(vertices_np)
    return vao, vbo, draw_commands

def main():
    if not glfw.init():
        print("Cannot initiate GLFW")
        return

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

    window = glfw.create_window(800, 600, "Motion Blur", None, None)
    if not window:
        glfw.terminate()
        print("Cannot create a window")
        return

    glfw.make_context_current(window)

    shader_program = create_shader_program("phong.vert", "phong.frag")
    glEnable(GL_DEPTH_TEST)
    if not shader_program:
        glfw.terminate()
        return

    # Load batched model
    car_vao, car_vbo, car_draw_commands = load_model("objects/Porsche_911_GT2.obj")
    road_vao, road_vbo, road_draw_commands = load_model("objects/straight_road.obj")
    pine_tree_left_vao, pine_tree_left_vbo, pine_tree_left_draw_commands = load_model("objects/pine_tree.obj")
    green_tree_right_vao, green_tree_right_vbo, green_tree_right_draw_commands = load_model("objects/green_tree.obj")

    glClearColor(0.1, 0.1, 0.1, 1.0)

    # Set up light and camera
    glUseProgram(shader_program)
    view = glm.lookAt(glm.vec3(0, 0, 2), glm.vec3(0, 0, 0), glm.vec3(0, 1, 0))
    fb_w, fb_h = glfw.get_framebuffer_size(window)
    projection = glm.perspective(glm.radians(45.0), fb_w / fb_h if fb_h != 0 else 4/3, 0.1, 100.0)

    def uni(name):
        loc = glGetUniformLocation(shader_program, name)
        if loc == -1:
            print(f"Warning: uniform '{name}' not found (location = -1). Maybe it's optimized out or name mismatch.")
        return loc

    def set_material(diffuse, specular, shininess):
        glUniform3f(uni("material.diffuse"), diffuse[0], diffuse[1], diffuse[2])
        glUniform3f(uni("material.specular"), specular[0], specular[1], specular[2])
        glUniform1f(uni("material.shininess"), shininess)
        return

    glUniform3f(uni("lightPos"), 1.2, 1.0, 2.0)
    glUniform3f(uni("viewPos"), 0.0, 0.0, 2.0)

    # Material
    glUniform3f(uni("material.ambient"), 0.2, 0.2, 0.2)
    glUniform3f(uni("material.diffuse"), 1.0, 0.5, 0.31)
    glUniform3f(uni("material.specular"), 0.5, 0.5, 0.5)
    glUniform1f(uni("material.shininess"), 32.0)

    # Light
    glUniform3f(uni("light.ambient"), 0.2, 0.2, 0.2)
    glUniform3f(uni("light.diffuse"), 0.8, 0.8, 0.8)
    glUniform3f(uni("light.specular"), 1.0, 1.0, 1.0)

    # Car
    car = Car(initial_position=glm.vec3(-0.1, -0.35, 0.0))
    previous_time = glfw.get_time() # Czas ostatniej klatki
    car.prev_position = glm.vec3(car.position)

    # Camera Mode
    camera_mode = 0  # 0: 3rd POV, 1: 1st POV
    def key_callback(window, key, scancode, action, mods):
        nonlocal camera_mode, height_offset
        if key == glfw.KEY_C and action == glfw.PRESS:
            camera_mode = 1 - camera_mode
    glfw.set_key_callback(window, key_callback)

    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    # Main loop
    while not glfw.window_should_close(window):
        # handle window resize
        fb_w, fb_h = glfw.get_framebuffer_size(window)
        glViewport(0, 0, fb_w, fb_h)
        # Update projection matrix on resize
        projection = glm.perspective(glm.radians(45.0), fb_w / fb_h if fb_h != 0 else 4/3, 0.1, 100.0)

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        time_val = glfw.get_time()
        delta_time = time_val - previous_time
        previous_time = time_val
        car.update(window, delta_time)

        # === CAMERA ===
        up = glm.vec3(0.0, 1.0, 0.0)
        forward_x, forward_z = -glm.sin(car.angle), -glm.cos(car.angle)
        forward_vector = glm.vec3(forward_x, 0.0, forward_z)

        if camera_mode == 0:
            backward_vector = -forward_vector
            distance_behind, height_above, look_ahead = 2.0, 0.5, 1.0
            eye = car.position + (backward_vector * distance_behind) + glm.vec3(0.0, height_above, 0.0)
            center = car.position + (forward_vector * look_ahead)
        else:
            right_vector = glm.vec3(-glm.cos(car.angle), 0.0, glm.sin(car.angle))
            height_offset, right_offset, forward_offset, look_ahead = 0.08, 0.06, -0.075, 1.0
            eye = car.position + glm.vec3(0.0, height_offset, 0.0) + (right_vector * right_offset) + (forward_vector * forward_offset)
            center = eye + (forward_vector * look_ahead)

        view = glm.lookAt(eye, center, up)

        # === CAR ===
        set_material([1.0, 0.1, 0.1], [0.9, 0.9, 0.9],  64.0)
        model_car = car.get_model_matrix()

        # model_car = glm.scale(model_car, glm.vec3(0.2))
        # model_car = glm.rotate(model_car, time_val * glm.radians(45.0), glm.vec3(0.0, 1.0, 0.0))
        # model_car = glm.translate(model_car, glm.vec3(-0.1, -0.35, 0.0))
        glUniform1f(glGetUniformLocation(shader_program, "u_alpha"), 1.0)
        render_complex_model(shader_program, car_vao, car_draw_commands, model_car, view, projection)

         # === LEFT TREE (pine) ===
        set_material([0.0, 0.4, 0.0], [0.2, 0.2, 0.2],  16.0)

        model_pine_tree_left = glm.mat4(1.0)
        model_pine_tree_left = glm.translate(model_pine_tree_left, glm.vec3(-0.7, -0.5, -1.0))
        # model_pine_tree_left = glm.rotate(model_pine_tree_left, time_val * glm.radians(45.0), glm.vec3(0.0, 1.0, 0.0))
        model_pine_tree_left = glm.scale(model_pine_tree_left, glm.vec3(0.15))
        render_complex_model(shader_program, pine_tree_left_vao, pine_tree_left_draw_commands, model_pine_tree_left, view, projection)

        # === RIGHT TREE (green) ===
        set_material([0.2, 0.8, 0.2], [0.3, 0.3, 0.3],  32.0)

        model_green_tree_right = glm.mat4(0.3)
        model_green_tree_right = glm.translate(model_green_tree_right, glm.vec3(0.5, -0.5, 0.0))
        # model_green_tree_right = glm.rotate(model_green_tree_right, time_val * glm.radians(45.0), glm.vec3(0.0, 1.0, 0.0))
        model_green_tree_right = glm.scale(model_green_tree_right, glm.vec3(0.2))
        render_complex_model(shader_program, green_tree_right_vao, green_tree_right_draw_commands, model_green_tree_right, view, projection)

        # === ROAD ===
        set_material([0.5, 0.5, 0.5], [0.1, 0.1, 0.1],  8.0)

        model_road = glm.mat4(1.0)
        model_road = glm.translate(model_road, glm.vec3(0.0, -0.5, 0.0))
        # model_road = glm.rotate(model_road, time_val * glm.radians(45.0), glm.vec3(0.0, 1.0, 0.0))
        model_road = glm.scale(model_road, glm.vec3(0.2))
        render_complex_model(shader_program, road_vao, road_draw_commands, model_road, view, projection)

        # ===MOTION BLUR ===
        scene_velocity = car.position - car.prev_position
        motion_blur_factor = 10
        blur_steps = 10

        for i in range(blur_steps):
            alpha = 1.0 / blur_steps
            step_fraction = i / blur_steps

            # ROAD
            distance_road = glm.length(car.position - glm.vec3(0.0, -0.5, 0.0))
            road_offset = scene_velocity * step_fraction * motion_blur_factor * (1.0 / distance_road)
            view_road_blur = glm.translate(view, -road_offset)
            glUniform1f(glGetUniformLocation(shader_program, "u_alpha"), alpha)
            set_material([0.5, 0.5, 0.5], [0.1, 0.1, 0.1],  8.0)
            render_complex_model(shader_program, road_vao, road_draw_commands, model_road, view_road_blur, projection)

            # LEFT TREE
            distance_pine = glm.length(car.position - glm.vec3(-0.7, -0.5, -1.0))
            pine_offset = scene_velocity * step_fraction * motion_blur_factor * (1.0 / distance_pine)
            view_pine_blur = glm.translate(view, -pine_offset)
            glUniform1f(glGetUniformLocation(shader_program, "u_alpha"), alpha)
            set_material([0.0, 0.4, 0.0], [0.2, 0.2, 0.2],  16.0)
            render_complex_model(shader_program, pine_tree_left_vao, pine_tree_left_draw_commands, model_pine_tree_left, view_pine_blur, projection)

            # RIGHT TREE
            distance_green = glm.length(car.position - glm.vec3(0.5, -0.5, 0.0))
            green_offset = scene_velocity * step_fraction * motion_blur_factor * (1.0 / distance_green)
            view_green_blur = glm.translate(view, -green_offset)
            glUniform1f(glGetUniformLocation(shader_program, "u_alpha"), alpha)
            set_material([0.2, 0.8, 0.2], [0.3, 0.3, 0.3],  32.0)
            render_complex_model(shader_program, green_tree_right_vao, green_tree_right_draw_commands, model_green_tree_right, view_green_blur, projection)

        glfw.swap_buffers(window)
        glfw.poll_events()

    glDeleteVertexArrays(1, [car_vao])
    glDeleteBuffers(1, [car_vbo])
    glDeleteVertexArrays(1, [road_vao])
    glDeleteBuffers(1, [road_vbo])
    glDeleteVertexArrays(1, [pine_tree_left_vao])
    glDeleteBuffers(1, [pine_tree_left_vbo])
    glDeleteVertexArrays(1, [green_tree_right_vao])
    glDeleteBuffers(1, [green_tree_right_vbo])

    glDeleteProgram(shader_program)

    glfw.terminate()


if __name__ == "__main__":
    main()
