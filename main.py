import glfw
from OpenGL.GL import *
import glm
import ctypes
from vertex_models import *


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

    # Main loop
    while not glfw.window_should_close(window):
        # handle window resize
        fb_w, fb_h = glfw.get_framebuffer_size(window)
        glViewport(0, 0, fb_w, fb_h)
        # Update projection matrix on resize
        projection = glm.perspective(glm.radians(45.0), fb_w / fb_h if fb_h != 0 else 4/3, 0.1, 100.0)


        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        time_val = glfw.get_time()

        # Render the batched car model
        model_car = glm.mat4(1.0)
        model_car = glm.translate(model_car, glm.vec3(-0.1, -0.35, 0.0))
        # model_car = glm.rotate(model_car, time_val * glm.radians(45.0), glm.vec3(0.0, 1.0, 0.0))
        model_car = glm.scale(model_car, glm.vec3(0.2))
        render_complex_model(shader_program, car_vao, car_draw_commands, model_car, view, projection)

        # Render the batched road model
        model_road = glm.mat4(1.0)
        model_road = glm.translate(model_road, glm.vec3(0.0, -0.5, 0.0))
        # model_road = glm.rotate(model_road, time_val * glm.radians(45.0), glm.vec3(0.0, 1.0, 0.0))
        model_road = glm.scale(model_road, glm.vec3(0.2))
        render_complex_model(shader_program, road_vao, road_draw_commands, model_road, view, projection)

        # Render the batched left pine tree model
        model_pine_tree_left = glm.mat4(1.0)
        model_pine_tree_left = glm.translate(model_pine_tree_left, glm.vec3(-0.7, -0.5, -1.0))
        # model_pine_tree_left = glm.rotate(model_pine_tree_left, time_val * glm.radians(45.0), glm.vec3(0.0, 1.0, 0.0))
        model_pine_tree_left = glm.scale(model_pine_tree_left, glm.vec3(0.15))
        render_complex_model(shader_program, pine_tree_left_vao, pine_tree_left_draw_commands, model_pine_tree_left, view, projection)

        # Render the batched right green tree model
        model_green_tree_right = glm.mat4(0.3)
        model_green_tree_right = glm.translate(model_green_tree_right, glm.vec3(0.5, -0.5, 0.0))
        # model_green_tree_right = glm.rotate(model_green_tree_right, time_val * glm.radians(45.0), glm.vec3(0.0, 1.0, 0.0))
        model_green_tree_right = glm.scale(model_green_tree_right, glm.vec3(0.2))
        render_complex_model(shader_program, green_tree_right_vao, green_tree_right_draw_commands, model_green_tree_right, view, projection)


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
