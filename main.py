import glfw
from OpenGL.GL import *
import numpy as np
import glm
import ctypes

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

    # vertices = [-0.5, -0.5, 0.0, 0.5, -0.5, 0.0, 0.0, 0.5, 0.0]
    vertices = np.array([
        #  pos               normal
        -0.5, -0.5, 0.0,     0.0, 0.0, 1.0,
         0.5, -0.5, 0.0,     0.0, 0.0, 1.0,
         0.0,  0.5, 0.0,     0.0, 0.0, 1.0,
    ], dtype=np.float32)

    # shader_program = create_shader_program("simple.vert", "simple.frag")
    shader_program = create_shader_program("phong.vert", "phong.frag")
    if not shader_program:
        glfw.terminate()
        return

    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)

    vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

    # glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
    # glEnableVertexAttribArray(0)
    stride = 6 * vertices.itemsize
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(12))
    glEnableVertexAttribArray(1)

    glBindBuffer(GL_ARRAY_BUFFER, 0)
    glBindVertexArray(0)

    # glClearColor(0.0, 0.0, 0.0, 1.0)
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
    loc_view = uni("view")
    loc_projection = uni("projection")
    loc_normalMatrix = uni("normalMatrix")
    loc_lightPos = uni("lightPos")
    loc_viewPos = uni("viewPos")
    # material
    loc_mat_ambient = uni("material.ambient")
    loc_mat_diffuse = uni("material.diffuse")
    loc_mat_specular = uni("material.specular")
    loc_mat_shin = uni("material.shininess")
    # light
    loc_light_ambient = uni("light.ambient")
    loc_light_diffuse = uni("light.diffuse")
    loc_light_specular = uni("light.specular")

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
        # glClear(GL_COLOR_BUFFER_BIT)

        # handle window resize
        fb_w, fb_h = glfw.get_framebuffer_size(window)
        glViewport(0, 0, fb_w, fb_h)

        # clear both color and depth
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # animate rotation so we actually see lighting change
        t = glfw.get_time()
        model = glm.rotate(glm.mat4(1.0), t * 0.8, glm.vec3(0.0, 1.0, 0.0))
        normalMatrix = glm.transpose(glm.inverse(glm.mat3(model)))

        if loc_model != -1:
            glUniformMatrix4fv(loc_model, 1, GL_FALSE, glm.value_ptr(model))
        if loc_normalMatrix != -1:
            glUniformMatrix3fv(loc_normalMatrix, 1, GL_FALSE, glm.value_ptr(normalMatrix))


        glUseProgram(shader_program)
        glBindVertexArray(vao)
        glDrawArrays(GL_TRIANGLES, 0, 3)
        glBindVertexArray(0)

        glfw.swap_buffers(window)
        glfw.poll_events()

    glDeleteVertexArrays(1, [vao])
    glDeleteBuffers(1, [vbo])
    glDeleteProgram(shader_program)

    glfw.terminate()


if __name__ == "__main__":
    main()
