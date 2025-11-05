import glfw
from OpenGL.GL import *
import numpy as np
import glm
import ctypes
import pywavefront


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

def cube_vertices():
    return np.array([
        # positions         # normals
        # front face
        -0.5, -0.5,  0.5,    0.0,  0.0,  1.0,
         0.5, -0.5,  0.5,    0.0,  0.0,  1.0,
         0.5,  0.5,  0.5,    0.0,  0.0,  1.0,
         0.5,  0.5,  0.5,    0.0,  0.0,  1.0,
        -0.5,  0.5,  0.5,    0.0,  0.0,  1.0,
        -0.5, -0.5,  0.5,    0.0,  0.0,  1.0,

        # back face
        -0.5, -0.5, -0.5,    0.0,  0.0, -1.0,
        -0.5,  0.5, -0.5,    0.0,  0.0, -1.0,
         0.5,  0.5, -0.5,    0.0,  0.0, -1.0,
         0.5,  0.5, -0.5,    0.0,  0.0, -1.0,
         0.5, -0.5, -0.5,    0.0,  0.0, -1.0,
        -0.5, -0.5, -0.5,    0.0,  0.0, -1.0,

        # left face
        -0.5,  0.5,  0.5,   -1.0,  0.0,  0.0,
        -0.5,  0.5, -0.5,   -1.0,  0.0,  0.0,
        -0.5, -0.5, -0.5,   -1.0,  0.0,  0.0,
        -0.5, -0.5, -0.5,   -1.0,  0.0,  0.0,
        -0.5, -0.5,  0.5,   -1.0,  0.0,  0.0,
        -0.5,  0.5,  0.5,   -1.0,  0.0,  0.0,

        # right face
         0.5,  0.5,  0.5,    1.0,  0.0,  0.0,
         0.5, -0.5, -0.5,    1.0,  0.0,  0.0,
         0.5,  0.5, -0.5,    1.0,  0.0,  0.0,
         0.5, -0.5, -0.5,    1.0,  0.0,  0.0,
         0.5,  0.5,  0.5,    1.0,  0.0,  0.0,
         0.5, -0.5,  0.5,    1.0,  0.0,  0.0,

        # bottom face
        -0.5, -0.5, -0.5,    0.0, -1.0,  0.0,
         0.5, -0.5, -0.5,    0.0, -1.0,  0.0,
         0.5, -0.5,  0.5,    0.0, -1.0,  0.0,
         0.5, -0.5,  0.5,    0.0, -1.0,  0.0,
        -0.5, -0.5,  0.5,    0.0, -1.0,  0.0,
        -0.5, -0.5, -0.5,    0.0, -1.0,  0.0,

        # top face
        -0.5,  0.5, -0.5,    0.0,  1.0,  0.0,
        -0.5,  0.5,  0.5,    0.0,  1.0,  0.0,
         0.5,  0.5,  0.5,    0.0,  1.0,  0.0,
         0.5,  0.5,  0.5,    0.0,  1.0,  0.0,
         0.5,  0.5, -0.5,    0.0,  1.0,  0.0,
        -0.5,  0.5, -0.5,    0.0,  1.0,  0.0
    ], dtype=np.float32)

def triangle_vertices():
    return np.array([
        #  pos               normal
        -0.5, -0.5, 0.0,     0.0, 0.0, 1.0,
         0.5, -0.5, 0.0,     0.0, 0.0, 1.0,
         0.0,  0.5, 0.0,     0.0, 0.0, 1.0,
    ], dtype=np.float32)


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

    # try:
    #     scene = pywavefront.Wavefront("car.obj", create_materials=True, parse=True)

    #     mesh = list(scene.meshes.values())[0]

    #     if not mesh.materials:
    #         raise Exception("cannot read vertices")

    #     material = mesh.materials[0]

    #     data = material.vertices
    #     vertex_format_string = material.vertex_format

    #     vertex_format_size = 0
    #     if "T2F" in vertex_format_string:
    #         vertex_format_size += 2
    #     if "T3F" in vertex_format_string:
    #         vertex_format_size += 3
    #     if "C3F" in vertex_format_string:
    #         vertex_format_size += 3
    #     if "N3F" in vertex_format_string:
    #         vertex_format_size += 3
    #     if "V3F" in vertex_format_string:
    #         vertex_format_size += 3

    #     if vertex_format_size == 0 or "V3F" not in vertex_format_string:
    #         raise Exception("Cannot find position data (V3F) in a model")

    #     positions = []
    #     for i in range(0, len(data), vertex_format_size):
    #         positions.append(data[i + vertex_format_size - 3])  # v_x
    #         positions.append(data[i + vertex_format_size - 2])  # v_y
    #         positions.append(data[i + vertex_format_size - 1])  # v_z

    #     vertices_np = np.array(positions, dtype=np.float32)
    #     vertex_count = len(positions) // 3

    # except Exception as e:
    #     print(f"Error reading the model: {e}")
    #     glfw.terminate()
    #     return

    # shader_program = create_shader_program("simple.vert", "simple.frag")
    shader_program = create_shader_program("phong.vert", "phong.frag")
    glEnable(GL_DEPTH_TEST)
    if not shader_program:
        glfw.terminate()
        return

    mvp_location = glGetUniformLocation(shader_program, "u_mvp")

    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)

    vertices_np = cube_vertices()

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

    glEnable(GL_DEPTH_TEST)
    glClearColor(0.1, 0.1, 0.1, 1.0)

    # Ustawienia światła i kamery
    glUseProgram(shader_program)
    model = glm.mat4(1.0)
    view = glm.lookAt(glm.vec3(0, 0, 2), glm.vec3(0, 0, 0), glm.vec3(0, 1, 0))
    projection = glm.perspective(glm.radians(45), 800/600, 0.1, 100.0)
    normalMatrix = glm.transpose(glm.inverse(glm.mat3(model)))

    def uni(name):
        loc = glGetUniformLocation(shader_program, name)
        if loc == -1:
            print(f"Warning: uniform '{name}' not found (location = -1). Maybe it's optimized out or name mismatch.")
        return loc

    loc_model = uni("model")
    loc_normalMatrix = uni("normalMatrix")

    fb_w, fb_h = glfw.get_framebuffer_size(window)

    # initial camera / projection
    view = glm.lookAt(glm.vec3(0.0, 0.0, 2.0), glm.vec3(0.0, 0.0, 0.0), glm.vec3(0.0, 1.0, 0.0))
    projection = glm.perspective(glm.radians(45.0), fb_w / fb_h if fb_h != 0 else 4/3, 0.1, 100.0)

    glUniformMatrix4fv(uni("model"), 1, GL_FALSE, glm.value_ptr(model))
    glUniformMatrix4fv(uni("view"), 1, GL_FALSE, glm.value_ptr(view))
    glUniformMatrix4fv(uni("projection"), 1, GL_FALSE, glm.value_ptr(projection))
    glUniformMatrix3fv(uni("normalMatrix"), 1, GL_FALSE, glm.value_ptr(normalMatrix))

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

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        projection = glm.perspective(glm.radians(45.0), 800 / 600, 0.1, 100.0)

        view = glm.lookAt(glm.vec3(0, 0, 5), glm.vec3(0, 0, 0), glm.vec3(0, 1, 0))

        model = glm.mat4(1.0)
        time_val = glfw.get_time()
        model = glm.rotate(model, time_val * glm.radians(50.0), glm.vec3(0.0, 1.0, 0.0))
        model = glm.scale(model, glm.vec3(0.2, 0.2, 0.2))
        normalMatrix = glm.transpose(glm.inverse(glm.mat3(model)))

        if loc_model != -1:
            glUniformMatrix4fv(loc_model, 1, GL_FALSE, glm.value_ptr(model))
        if loc_normalMatrix != -1:
            glUniformMatrix3fv(loc_normalMatrix, 1, GL_FALSE, glm.value_ptr(normalMatrix))

        mvp = projection * view * model

        glUseProgram(shader_program)

        glUniformMatrix4fv(mvp_location, 1, GL_FALSE, glm.value_ptr(mvp))

        glBindVertexArray(vao)

        # TODO
        vertex_count = 36

        glDrawArrays(GL_TRIANGLES, 0, vertex_count)

        glBindVertexArray(0)

        glfw.swap_buffers(window)
        glfw.poll_events()

    glDeleteVertexArrays(1, [vao])
    glDeleteBuffers(1, [vbo])
    glDeleteProgram(shader_program)

    glfw.terminate()


if __name__ == "__main__":
    main()
